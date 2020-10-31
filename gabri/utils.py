import inspect
import math
import shutil
import time
import copy
import collections
import os
import sys
import datetime

import PIL
import numpy as np
import matplotlib.pyplot as plt
import random
import re

import torch
import unicodedata
from PIL import Image
from scipy.stats import pearsonr
from scipy import stats


def get_model_path_name(args, args_name, folder, main_classes, extension='.pth'):
    arguments = collections.OrderedDict()
    for arg in args_name:
        arguments[arg] = args[arg]
    path_name = str([item for item in arguments.items()])
    if main_classes:
        path_name += "_main_classes"
    save_path = sanitize_filename(path_name, folder=folder, extension=extension)
    return save_path


def sanitize_filename(value: str, max_chars: int = 500, extension: str = None, allow_unicode: bool = False,
                      folder: str = None):
    """
	Convert to ASCII if 'allow_unicode' is False. Convert spaces to hyphens.
	Remove characters that aren't alphanumerics, underscores, or hyphens.
	Convert to lowercase. Also strip leading and trailing whitespace.
	"""

    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    filename = re.sub(r'[-\s]+', '-', value)

    if extension is not None:
        extension = extension[1:] if extension.startswith(".") else extension
        max_chars = max_chars - (len(extension) + 1)

    if folder:
        filename = os.path.join(folder, filename)
    filename = filename[:max_chars]
    if extension:
        filename = filename + "." + extension

    return filename


def sample_weights_semisup_learning(n_sup_per_class: int, labels: list):
    active_sample_per_class = [np.nonzero(labels[:, i])[0].tolist() for i in range(labels.shape[1])]
    active_sample_lenghts = [len(active_samples) for active_samples in active_sample_per_class]

    assert np.asarray([length > n_sup_per_class for length in active_sample_lenghts]).all(), \
        "Number of supervision per class %d is too high. max number" \
        "of supervision allowed: %d" % (n_sup_per_class, min(active_sample_lenghts))

    semi_sup_index_list_per_class = [random.sample(active_sample_per_class[i], n_sup_per_class)
                                     for i in range(labels.shape[1])]
    semi_sup_indexes = np.unique(np.concatenate(semi_sup_index_list_per_class))
    sample_weights = np.asarray([1 if i in semi_sup_indexes else 0
                                 for i in range(labels.shape[0])])

    return sample_weights


def show_batch(image_batch, label_batch, labels_names, save=False):
    fig = plt.figure()
    batch_size = image_batch.shape[0] if isinstance(image_batch, torch.Tensor) or isinstance(image_batch, np.ndarray) else len(image_batch)
    for n in range(batch_size):
        ax = plt.subplot(np.sqrt(batch_size), np.sqrt(batch_size), n + 1)
        sample = image_batch[n]
        if isinstance(sample, torch.Tensor):
            sample = sample.numpy()
        elif isinstance(sample, np.ndarray):
            sample = sample.squeeze()
        elif isinstance(sample, Image.Image):
            sample = np.asarray(sample)
        else:
            raise NotImplementedError
        if sample.shape[0] == 3:
            sample = np.rollaxis(sample, 0, 3)
        if sample.max() < 200 or sample.min() < 0:
            sample = (sample * 255).astype(np.uint8)
        # im = Image.fromarray(sample)
        # im.show()
        plt.imshow(sample)
        title = ""
        label = label_batch[n]
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        label = label.squeeze()
        for i in np.where(label == 1)[0]:
            title += labels_names[i] + ", "
        title = title[:40]
        ax.set_title(title)
        ax.axis('off')
    if save:
        plt.savefig("../images/fig" + str(np.random.randint(1000)))
    plt.show()


class Logger(object):
    def __init__(self, verbose=1):
        self._verbose = verbose
        if self._verbose:
            self.terminal = sys.stdout
        self.log = open("logfile.log", "a")

    def write(self, message):
        if self._verbose:
            self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        if self._verbose:
            self.terminal.flush()
        self.log.flush()
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def plot_metrics(metrics: list, metric_name: str = "Metrics", title: str = None, exp_names=None, epochs=None):
    lines = []
    if epochs is None:
        epochs = len(metrics[0])

    if title is None:
        title = "Metrics Comparison"

    for i in range(len(metrics)):
        lines.append(plt.plot(range(0, epochs), metrics[i], linewidth=3.0)[0])

    if exp_names is not None:
        plt.legend(lines, exp_names)
    plt.ylabel(metric_name)
    plt.xlabel('Epochs')
    if title is not None:
        plt.title(title)
    plt.savefig(title + '.png')

    plt.show()
    plt.clf()


def get_unique_filename(save_path):
    i = 2
    filename, ext = os.path.splitext(save_path)
    while os.path.isfile(save_path):
        save_path = filename + str(i) + ext
        i += 1
    return save_path


def find_folder(folder: str = "models", n: int = 10, absolute_path: bool = False):
    for i in range(n):
        if os.path.isdir(folder):
            if absolute_path:
                folder = os.path.abspath(folder)
            return folder
        else:
            folder = os.path.join("..", folder)
    raise FileNotFoundError(str(folder) + " folder not found")


def explore_params(solutions: list, solution: dict, params: dict):
    if not params:
        solutions.append(copy.deepcopy(solution))
        return
    else:
        param_item = params.popitem()
        for param in param_item[1]:
            solution[param_item[0]] = param
            explore_params(solutions, solution, params)
            solution.pop(param_item[0])
        params[param_item[0]] = param_item[1]


def switch_to_exp_dir(exp_dir: str, result_dir=None, father_dir: str = None):
    if father_dir is None:
        father_dir = "results"
    if result_dir:
        if not os.path.isdir(result_dir):
            try:
                result_dir = find_folder(result_dir)
            except BaseException:
                try:
                    res_dir = find_folder(father_dir)
                except BaseException:
                    res_dir = father_dir
                    os.mkdir(res_dir)
                if not os.path.isdir(os.path.join(res_dir, result_dir)):
                    os.mkdir(os.path.join(res_dir, result_dir))
                result_dir = os.path.join(res_dir, result_dir)
    else:
        try:
            result_dir = find_folder("results")
        except BaseException:
            result_dir = "../results"
            os.mkdir(result_dir)
    os.chdir(result_dir)

    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    os.chdir(exp_dir)


def grid_search(fun, key_metric: str, params: dict, extra_params=None, result_dir=None, father_dir=None):
    if extra_params is None:
        extra_params = {}

    switch_to_exp_dir(result_dir=result_dir, father_dir=father_dir,
                      exp_dir=sanitize_filename("grid on " + str([key for key in params.keys()])))

    list_parameters = []
    explore_params(list_parameters, {}, params=params)
    print("Exploration space dimension: %d" % (len(list_parameters)))

    # Calculate solutions
    solutions = {}
    best_solution = ()
    for i, param in enumerate(list_parameters):
        # if i == 0: continue
        print("\n\nSolution %d/%d. \nParams: %s" % (i + 1, len(list_parameters), param))
        run_dir = sanitize_filename(str(param))

        # Check whether the solution has already been calculated and calculation completed
        if os.path.isdir(run_dir) and len(os.listdir(run_dir)) > 0 and os.path.isfile(
                os.path.join(run_dir, "mean_results.txt")):
            mean_metrics, sem_metrics = {}, {}
            results_file = os.path.join(run_dir, "mean_results.txt")
            with open(results_file) as f:
                results_lines = f.readlines()[2:]
                for line in results_lines:
                    if len(line.split(" ")) > 1:
                        metric_name = line.split(" ")[1]
                        metric_mean = eval(line.split(" ")[3]) if line.split(" ")[3] != "nan" else 0
                        metric_sem = eval(line.split(" ")[5]) if line.split(" ")[5] != "nan" else 0
                        mean_metrics[metric_name] = metric_mean
                        sem_metrics[metric_name] = metric_sem
        else:
            if os.path.isdir(run_dir):
                shutil.rmtree(run_dir, ignore_errors=False)
            if not is_pathname_valid(run_dir):
                raise (FileNotFoundError())

            os.mkdir(run_dir)
            extra_params['result_dir'] = run_dir

            fun_params = {**param, **extra_params}

            mean_metrics, sem_metrics = fun(**fun_params)

        param = str(param)
        solutions[param] = (mean_metrics, sem_metrics)

    # Calculate best solution
    best_mean_metric = 0 if ("loss" not in key_metric and "n_wrong" not in key_metric) else sys.float_info.max
    for param, metrics in solutions.items():
        mean_metrics = metrics[0]
        sem_metrics = metrics[1]
        if ("loss" not in key_metric and "n_wrong" not in key_metric) and best_mean_metric < mean_metrics[key_metric] or \
                ("loss" in key_metric or "n_wrong" in key_metric) and best_mean_metric > mean_metrics[key_metric]:
            best_solution = (param, (mean_metrics, sem_metrics))
            best_mean_metric = mean_metrics[key_metric]

    # Calculate Correlation between parameters values and the key metrics
    params_values = {}
    key_metric_values = []
    params_correlations = {}
    for solution_params, solution_metrics in solutions.items():
        key_metric_values.append(solution_metrics[0][key_metric])
        dict_params = eval(solution_params)
        for param, value in dict_params.items():
            if isinstance(value, list) and len(value) > 0 and (
                    isinstance(value[0], float) or isinstance(value[0], int)):
                value = np.sum(value)
            elif isinstance(value, list) and len(value) == 0:
                value = 0
            elif not (isinstance(value, float) or isinstance(value, int)):
                continue
            if param in params_values:
                params_values[param].append(value)
            else:
                params_values[param] = [value]
    if len(list_parameters) > 1:
        for param_name, values in params_values.items():
            params_correlations[param_name] = pearsonr(key_metric_values, values)[0]

    # Save results on a file
    j = 2
    f_name = "grid_search_results.txt"
    if os.path.isfile(f_name):
        while os.path.isfile(f_name[:-4] + str(j) + ".txt"):
            j = j + 1
        f_name = f_name[:-4] + str(j) + ".txt"
    with open(f_name, 'w') as f:
        # Save Best Solution
        best_solution_param = best_solution[0]
        f.write("Best solution params: %s\n" % best_solution_param)
        best_solution_mean_metrics: dict = best_solution[1][0]
        best_solution_sem_metrics: dict = best_solution[1][1]
        f.write("Best solution metrics: ")
        for metric_name, metric_mean in best_solution_mean_metrics.items():
            f.write("%s Mean = %.4f +- %.4f\t" % (metric_name, metric_mean, best_solution_sem_metrics[metric_name]))
        f.write("\r \n")

        # Save Parameters Correlations
        if len(list_parameters) > 1:
            f.write("Parameters Correlation with the key metric %s:\n" % key_metric)
            for param_name, corr in params_correlations.items():
                f.write("%s parameter: %.4f\t" % (param_name, corr))
            f.write("\r \n")

        # Save Parameters Performances
        for i, solution in enumerate(solutions.items()):
            solution_param = solution[0]
            f.write("%d) Solution params: %s\n" % (i, solution_param))
            solution_mean_metrics: dict = solution[1][0]
            solution_sem_metrics: dict = solution[1][1]
            f.write("Solution metrics: ")
            for metric_name, metric_mean in solution_mean_metrics.items():
                f.write("%s Mean = %.4f +- %.4f\t" % (metric_name, metric_mean, solution_sem_metrics[metric_name]))
            f.write("\r \n")


def parallel_runner(fun, n_run=1, result_dir=None, logging=True, verbose=1, max=True, **fun_kwargs, ):
    if not result_dir:
        result_dir = sanitize_filename(str(fun_kwargs) + "-time-" + str(datetime.datetime.now()))
    shutil.rmtree(result_dir[:255], ignore_errors=False) if os.path.isdir(result_dir[:255]) else 0
    os.mkdir(result_dir[:255])
    os.chdir(result_dir[:255])

    # Dictionary of lists of the various metric values over the various runs
    diff_run_metrics = {}
    diff_run_trends = {}
    flag = False

    for z in range(n_run):

        print("\nRun %d/%d\n" % (z + 1, n_run))
        # Create and change directory
        os.mkdir(str(z))
        os.chdir(str(z))

        # Duplicate logging: terminal and log file. TODO: try worksheet logger
        original = sys.stdout
        if logging: sys.stdout = Logger(verbose=verbose)

        # Set seed according to the run if not already fixed
        if 'seed' in inspect.getfullargspec(fun)[0] and ('seed' not in fun_kwargs or flag):
            fun_kwargs["seed"] = z
            flag = True

        # Run Function
        metrics_list = fun(**fun_kwargs)
        # metrics_list = {"train_acc":[0,1,2],"test_acc":[1,3,4],"val_acc":[1,1,3],} # TO DEBUG

        # Check returned metrics to be collected in a dictionary
        assert isinstance(metrics_list, dict), "Wrong returned metrics"

        # Get maximum values of the metric
        metrics = {}
        bias = 0  # len(list(metrics_list.items())[1][1]) // 4
        for (metric_name, metric_list) in metrics_list.items():
            if hasattr(metric_list, '__len__'):
                if "loss" in metric_name:
                    metrics[metric_name] = np.min(metric_list[bias:]) if max else metric_list[-1]
                elif "rules" in metric_name:
                    metrics[metric_name] = metric_list
                else:
                    metrics[metric_name] = np.max(metric_list[bias:]) if max else metric_list[-1]
            else:
                metrics[metric_name] = metric_list

        # Write max results on text file and on terminal
        with open("results.txt", 'w') as f:
            f.write("Experiment Number: %d\nComplete Folder Name: %s\nParameters: %s\r\n" % (
                z, result_dir, str(fun_kwargs)))
            for metric_name, metric in metrics.items():
                if "rules" in metric_name:
                    f.write(metric_name + "\r\n")
                    for i, rule in enumerate(metric):
                        f.write(str(i) + ") %s\r\n" % (metric))
                else:
                    f.write(metric_name + ": %.4f\r\n" % (metric))

        # Append max results to list of max results in the various running
        for metric_name, metric in metrics.items():
            if isinstance(metric, str):
                continue
            if z == 0:
                diff_run_metrics[metric_name] = []
                diff_run_trends[metric_name] = []
            diff_run_metrics[metric_name].append(metric)
            diff_run_trends[metric_name].append(metrics_list[metric_name])

        # Reverse previous actions
        sys.stdout = original
        os.chdir("..")

    # Plot metrics for which we have an history on a graph
    for metric_name, metric_list in diff_run_trends.items():
        if hasattr(metric_list[0], '__len__'):
            plot_metrics(metric_list, metric_name=metric_name, title=metric_name + " over different run",
                         exp_names=range(len(metric_list)))

    # Compute mean and mean standard error
    mean_metrics = {}
    sem_metrics = {}
    for metric_name, metric_values in diff_run_metrics.items():
        mean_metrics[metric_name] = np.mean(metric_values)
        sem_metrics[metric_name] = stats.sem(metric_values) if not math.isnan(stats.sem(metric_values)) else 0

    # Write results on terminal and on disk
    with open("mean_results.txt", 'w') as f:
        f.write("Folder Name : %s\nParameters: %s\n" % (result_dir, str(fun_kwargs)))
        for metric_name, metric_mean in mean_metrics.items():
            f.write("Mean %s = %.4f +- %.4f \r\n" % (metric_name, metric_mean, sem_metrics[metric_name]))

    os.chdir("..")
    trend_metrics = diff_run_trends

    return mean_metrics, sem_metrics  # , trend_metrics


def file_exists(filename: str):
    cur_dir = os.path.abspath(os.curdir)
    os.chdir(os.path.dirname(filename))
    result = os.path.isfile(os.path.basename(filename))
    os.chdir(cur_dir)
    return result

# import openpyxl as xls
# class Worksheet_logger:
#
# 	def __init__(self, filename, params, metrics, reuse=True):
#
# 		self.params			= params
# 		self.metrics		= metrics
# 		self.header			= [params, "Epochs", metrics]
# 		self.column_dict 	= {}
# 		self.column_dict.update({value: key} for value, key in enumerate(self.header))
# 		self.n_row 			= 1
# 		self.solutions:dict = {}
# 		self.filename 		= filename
#
# 		if not os.path.isfile(filename):
#
# 			self.wb = xls.Workbook()
# 			self.wb.save(filename)
# 			self.ws = self.wb.active
# 			# Append header with
# 			self.ws.append([params, metrics])
# 			self.wb.save(filename)
# 			print("Created new worbook")
#
# 		else:
# 			if reuse:
# 				self.wb = xls.load_workbook(filename=filename)
# 				self.ws = self.wb.active
# 				print("Reopening already existing workbook")
# 				# Reading the header
# 				r = self.ws.rows[0]
# 				existing_header = [cell.value for cell in r]
# 				if existing_header != self.params:
# 					raise("Inconsistent header found: header in the worksheet: %s, header given %s"
# 					      %(existing_header, self.header))
# 				# Iterate over rows, skipping header row
# 				for r in self.ws.iter_rows(min_row=2):
# 					individual = {(self.column_dict[i], cell.value) for i, cell in enumerate(r)}
# 					self.solutions[self.n_row] = individual
# 					self.n_row += 1
# 				print("Number of row read", self.n_row)
# 			else:
# 				raise (Exception("Already exsiting workbook"))
#
# 		return
#
# 	def writeRow(self, params:dict, metrics:dict, epoch:int):
# 		for param in params.keys():
# 			if param not in self.params:
# 				print("Cannot write row, params inconsistent")
# 				return
#
# 		for metric in metrics.keys():
# 			if metric not in self.metrics:
# 				print("Cannot write row, metrics inconsistent")
# 				return
#
# 		new_row = self.ws[self.n_row + 1]
# 		for (param, value) in params:
# 			new_row[self.column_dict[param]] = value
#
# 		for (metric, value) in metrics:
# 			new_row[self.column_dict[metric]] = value
#
# 		new_row[self.column_dict["Epoch"]] = epoch
#
# 		self.n_row += 1
# 		self.wb.save(self.filename)
#
# 		return self.n_row
#
# 	def updateRow(self, row, metrics:dict, epoch:int):
# 		for metric in metrics.keys():
# 			if metric not in self.metrics:
# 				print("Cannot write row, metrics inconsistent")
# 				return
#
# 		row_to_update = self.ws[row]
#
# 		for (metric, value) in metrics:
# 			row_to_update[self.column_dict[metric]] = value
#
# 		row_to_update[self.column_dict["Epoch"]] = epoch
# 		self.wb.save(self.filename)
# 		return


class Progbar(object):
    """Displays a progress bar.

	# Arguments
		target: Total number of steps expected, None if unknown.
		width: Progress bar width on screen.
		verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
		stateful_metrics: Iterable of string names of metrics that
			should *not* be averaged over time. Metrics in this list
			will be displayed as-is. All others will be averaged
			by the progbar before display.
		interval: Minimum visual progress update interval (in seconds).
	"""

    def __init__(self, target, width=30, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        self._values = collections.OrderedDict()
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.

		# Arguments
			current: Index of current step.
			values: List of tuples:
				`(name, value_for_last_step)`.
				If `name` is in `stateful_metrics`,
				`value_for_last_step` will be displayed as-is.
				Else, an average of the metric over time will be displayed.
		"""
        values = values or []
        for k, v in values:
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                # Stateful metrics output a numeric value.  This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            # if self._dynamic_display:
            sys.stdout.write('\b' * prev_total_width)
            sys.stdout.write('\r')
            # else:
            # 	sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = ('%d:%02d:%02d' %
                                  (eta // 3600, (eta % 3600) // 60, eta % 60))
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values:
                    info += ' - %s:' % k
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)


import errno
ERROR_INVALID_NAME = 123


def is_pathname_valid(pathname: str) -> bool:
    """
    `True` if the passed pathname is a valid pathname for the current OS;
    `False` otherwise.
    """
    # If this pathname is either not a string or is but is empty, this pathname
    # is invalid.
    try:
        if not isinstance(pathname, str) or not pathname:
            return False

        # Strip this pathname's Windows-specific drive specifier (e.g., `C:\`)
        # if any. Since Windows prohibits path components from containing `:`
        # characters, failing to strip this `:`-suffixed prefix would
        # erroneously invalidate all valid absolute Windows pathnames.
        _, pathname = os.path.splitdrive(pathname)

        # Directory guaranteed to exist. If the current OS is Windows, this is
        # the drive to which Windows was installed (e.g., the "%HOMEDRIVE%"
        # environment variable); else, the typical root directory.
        root_dirname = os.environ.get('HOMEDRIVE', 'C:') \
            if sys.platform == 'win32' else os.path.sep
        assert os.path.isdir(root_dirname)   # ...Murphy and her ironclad Law

        # Append a path separator to this directory if needed.
        root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep

        # Test whether each path component split from this pathname is valid or
        # not, ignoring non-existent and non-readable path components.
        for pathname_part in pathname.split(os.path.sep):
            try:
                os.lstat(root_dirname + pathname_part)
            # If an OS-specific exception is raised, its error code
            # indicates whether this pathname is valid or not. Unless this
            # is the case, this exception implies an ignorable kernel or
            # filesystem complaint (e.g., path not found or inaccessible).
            #
            # Only the following exceptions indicate invalid pathnames:
            #
            # * Instances of the Windows-specific "WindowsError" class
            #   defining the "winerror" attribute whose value is
            #   "ERROR_INVALID_NAME". Under Windows, "winerror" is more
            #   fine-grained and hence useful than the generic "errno"
            #   attribute. When a too-long pathname is passed, for example,
            #   "errno" is "ENOENT" (i.e., no such file or directory) rather
            #   than "ENAMETOOLONG" (i.e., file name too long).
            # * Instances of the cross-platform "OSError" class defining the
            #   generic "errno" attribute whose value is either:
            #   * Under most POSIX-compatible OSes, "ENAMETOOLONG".
            #   * Under some edge-case OSes (e.g., SunOS, *BSD), "ERANGE".
            except OSError as exc:
                if hasattr(exc, 'winerror'):
                    if exc.winerror == ERROR_INVALID_NAME:
                        return False
                elif exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                    return False
    # If a "TypeError" exception was raised, it almost certainly has the
    # error message "embedded NUL character" indicating an invalid pathname.
    except TypeError as exc:
        return False
    # If no exception was raised, all path components and hence this
    # pathname itself are valid. (Praise be to the curmudgeonly python.)
    else:
        return True
    # If any other exception was raised, this is an unrelated fatal issue
    # (e.g., a bug). Permit this exception to unwind the call stack.
    #
    # Did we mention this should be shipped with Python already?