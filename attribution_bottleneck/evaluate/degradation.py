from collections import OrderedDict
from datetime import datetime
from math import sqrt

from tqdm import tqdm

from attribution.base import *
from evaluate.base import *
from attribution_bottleneck.evaluate.perturber import *
from utils.baseline import Mean
from utils.misc import *


class DegradationEval(Evaluation):
    """ Evaluates single heatmaps by perturbing the image on high rated regions """
    # TODO pass BATCHES with perturbed images

    def __init__(self, model, jump=2, part=1., baseline=None, tile_size=None, reverse=False, eval_mode="probs"):
        self.model = model
        self.show_original = False
        self.show_heatmap = False
        self.show_order = False
        self.show_baseline = False
        self.show_lowest_score = False
        self.show_step = None
        self.progbar = False
        self.warn_baseline_prob = 0.05
        self.reverse = reverse
        self.part = part
        self.jump = jump
        self.tile_size = tile_size
        self.eval_mode = eval_mode
        self.baseline = baseline if baseline is not None else Mean()

        class SoftMaxWrapper(nn.Module):
            def __init__(self, logits):
                super().__init__()
                self.logits = logits
                self.softmax = nn.Softmax(dim=1)

            def forward(self, input):
                return self.softmax(self.logits(input))
        if self.eval_mode == "logits":
            pass
        elif self.eval_mode == "probs":
            self.model = SoftMaxWrapper(self.model)
        else:
            raise ValueError

    def eval(self, hmap: np.ndarray, img_t: torch.Tensor) -> dict:
        """
        Iteratively perturbe an image, measure classification drop, and return the path of ([num perturbations] -> [class drop]) tuples
        TODO do this all in torch - it may be faster
        """
        assert img_t.shape[0] == 1, "batch dimension has to be one - we analyze one input sample"
        assert img_t.shape[1] == 3, "RGB images required"
        assert isinstance(img_t, torch.Tensor), "img_t has to be a torch tensor"
        assert isinstance(hmap, np.ndarray), "the heatmap has to be a np.ndarray"
        assert hmap.shape == tuple(img_t[0,0].shape), "heatmap and image have to be the same size: {} != {}".format(hmap.shape, tuple(img_t[0,0].shape))

        self.model.eval()

        # construct the perturbed image
        img = to_np_img(img_t)
        baseline_img = self.baseline.apply(img)
        baseline_t = to_img_tensor(baseline_img, device=img_t.device)

        # calculate intial score and baseline
        initial_out = self.eval_np(img_t)
        eval_point = np.argmax(initial_out)
        initial_val = initial_out[eval_point]
        baseline_val = self.eval_np(baseline_t)[eval_point]
        if baseline_val > self.warn_baseline_prob:
            print("Warning: score is still {}".format(baseline_val))
            show_img(denormalize(baseline_img))

        # process heatmap
        perturber = PixelPerturber(img_t, baseline_img) if (self.tile_size is None or self.tile_size == (1, 1)) else GridPerturber(img_t, baseline_t, self.tile_size)
        idxes = perturber.get_idxes(hmap, reverse=self.reverse)

        # iterate with perturbing
        max_steps = len(idxes)
        do_steps = int(max_steps * self.part)
        results = np.full((do_steps+1, 2), 0.0)
        min_value = initial_val
        min_degraded_t = img_t
        for step in tqdm(range(do_steps), desc="Perturbing", disable=not self.progbar):

            # perturbe (in numpy)
            perturber.perturbe(*idxes[step])
            perturbed_t = perturber.get_current()
            part = (step+1)/max_steps
            if not step % self.jump == 0:
                continue

            # pass thru (as tensor)
            value = self.eval_np(perturbed_t)[eval_point]
            results[step+1:step+self.jump+2] = [part, value - initial_val]
            if value < min_value:
                min_value = value
                min_degraded_t = perturbed_t

            # maybe show current step
            if self.show_step is not None and step % self.show_step == 0:
                val = self.eval_np(perturbed_t)[eval_point]
                show_img(perturbed_t, title=str(val))

        # maybe show results
        if self.show_original:
            show_img(normalize_img(img), title="original image, score {}".format(initial_val))
        if self.show_baseline:
            show_img(normalize_img(baseline_img), title="baseline image, score {}".format(baseline_val))
        if self.show_heatmap:
            show_img(normalize_img(hmap), title="heatmap")
        if self.show_heatmap and isinstance(perturber, GridPerturber):
            show_img(normalize_img(GridView(np.expand_dims(hmap, 2), *perturber.get_tile_shape()).get_tile_means()), title="heatmap grid")
        if self.show_order:
            order_map = np.zeros(perturber.get_grid_shape())
            for i, idx in enumerate(idxes):
                order_map[idx] = -i
            show_img(normalize_img(order_map), title="order of perturbation, w=first")
        if self.show_lowest_score:
            show_img(normalize_img(to_np_img(min_degraded_t)), title="lowest score of: {:.3f} ({:.3f})".format(min_value, min_value - initial_val))

        return {
            "path": results,
        }

    def eval_np(self, img_t):
        """ pass the tensor through the network and return the scores as a numpyarray w/o batch dimension (1D shape)"""
        return to_np(self.model(img_t))[0]

class SensitivityN(Evaluation):
    pass

class Collector:
    """ Use a evaluator+samples to evaluate multiple methods """

    def __init__(self, evaluator, methods):
        self.evaluator = evaluator
        self.method_settings = methods
        self.show_heatmap = False
        self.progbar = True
        self.device = next(iter(evaluator.model.parameters())).device

    def make_eval(self, samples: list):
        assert isinstance(samples, list)
        assert len(samples) > 0
        assert isinstance(samples[0], tuple)
        assert isinstance(samples[0][0], torch.Tensor)  # img_t
        assert isinstance(samples[0][1], torch.Tensor)  # target_t

        # Collect results for each method
        total_ms = {name: 0.0 for name in self.method_settings}
        paths = {name: [] for name in self.method_settings}
        for sample in tqdm(samples, desc="Evaluating", disable=not self.progbar):
            for name, meth in self.method_settings.items():
                start = datetime.now()
                sample = sample[0].to(self.device), sample[1].to(self.device)
                hmap = meth.heatmap(input_t=sample[0], target_t=sample[1])
                total_ms[name] += (datetime.now() - start).total_seconds()
                if self.show_heatmap:
                    show_img(hmap)

                result = self.evaluator.eval(hmap, sample[0].clone())
                paths[name].append(result["path"])

        return {name: {
            "paths": np.stack([path for path in paths[name]]),
            "time_ms": int(total_ms[name] * 1000.0 / len(paths[name])),
        } for name in self.method_settings}


class GraphDrawer:
    """ Draws the evaluation of multiple methods over multiple images """

    def __init__(self, figsize=(18, 10)):
        self.figsize = figsize
        self.show_title = True
        self.show_times = False

    @staticmethod
    def integrals(results):
        diffs = {k: results[0][k]["paths"].mean(0) - results[1][k]["paths"].mean(0) for k in results[0]}
        return {k: diffs[k][:,1].mean() for k in diffs}

    def draw_fig(self, method_results: dict, *args, **kwargs):
        plt.figure()
        _, ax = plt.subplots(figsize=self.figsize, dpi=80)
        self.draw(method_results, ax=ax, *args, **kwargs)
        n = next(iter(method_results.values()))["paths"].shape[0]
        plt.ylabel("Drop in accuracy over {} samples".format(n))
        plt.xlabel("Proportion of input replaced")
        plt.title("Comparison over attribution methods for {} samples".format(n))
        plt.show()

    def draw(self, method_results: dict, ax, mode="mean-std", colors=None, lines=None, labels=None, order=None):

        n = next(iter(method_results.values()))["paths"].shape[0]

        # Make style arguments
        kwargs = [{} for _ in method_results]  # Initialize empty
        colors = colors if colors else (plt.get_cmap("tab10").colors if len(method_results) < 9 else plt.get_cmap("tab20").colors)  # Default color map
        if lines is not None:
            for i, _ in enumerate(method_results):
                kwargs[i]["linestyle"] = lines[i]
        if labels is not None:
            assert len(labels) == len(method_results)
            for i, name in enumerate(method_results):
                print(f"setting {name}  -> {labels[i]}")
                kwargs[i]["label"] = labels[i]
        else:
            for i, name in enumerate(method_results):
                kwargs[i]["label"] = name

        for kwarg in kwargs:
            kwarg["label"] = kwarg["label"].replace("_", "\_") if kwarg["label"] is not None else kwarg["label"]

        # Remove ignored lines
        if order is not None:
            keys = list(method_results.keys())
            vals = list(method_results.values())
            method_results = OrderedDict({keys[o]: vals[o] for o in order})
            kwargs = [kwargs[o] for o in order]

        # Assign colors
        if colors is not None:
            for i, _ in enumerate(method_results):
                kwargs[i]["color"] = colors[i]

        # Plot results
        # Paths are [sample, blur_steps, prob_steps]  (blur step is a bit redundant)
        if mode == "mean-std":
            for i, (name, result) in enumerate(method_results.items()):
                paths = method_results[name]["paths"]
                xm, xs = paths.mean(axis=0), paths.std(axis=0) / sqrt(n)
                m, s = xm[:,1], xs[:,1]
                x = xm[:,0]
                p = ax.plot(x, m, **kwargs[i])[0]
                ax.fill_between(x, m + s, m - s, color=p.get_color(), alpha=0.2)

        elif mode == "mean":
            for i, (name, result) in enumerate(method_results.items()):
                paths = method_results[name]["paths"]
                xm = paths.mean(axis=0)
                m = xm[:,1]
                x = xm[:,0]
                ax.plot(x, m, **kwargs[i])

        elif mode == "all":
            colors = plt.cm.rainbow(np.linspace(0,1,len(method_results)))
            for meth_i, (name, result) in enumerate(method_results.items()):
                for path_i, path in enumerate(method_results[name]["paths"]):
                    kwargs[meth_i]["label"] = kwargs[meth_i]["label"] if path_i == 0 else None
                    kwargs[meth_i]["color"] = colors[meth_i]
                    ax.plot(path, **kwargs[meth_i])
        else:
            raise ValueError

        ax.legend(prop={'size': 9}, loc=1)

        # print timing
        if self.show_times:
            for i, (name, result) in enumerate(method_results.items()):
                print("{}: {}ms".format(name, method_results[name]["time_ms"]))

