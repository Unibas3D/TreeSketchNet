# BSD 3-Clause License

# Copyright (c) 2021, ...
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import re
import tensorflow as tf
from write_logs import write_img_logs
import numpy as np
import signal
import os
import sys
import time
import tensorflow_addons as tfa
from collections import defaultdict


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def add_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def dataset_info(full_dataset, print_description):
        cardinality = tf.data.experimental.cardinality(full_dataset).numpy()
        if print_description is True:
            print("Cardinality: ", cardinality)
            print("Dataset shape: ", full_dataset.element_spec)
            for image, label in full_dataset.take(1):
                print("Dataset img shape: ", image.shape)
                print("Dataset label shape: ", label.shape)
        return cardinality

class StopCallback(tf.keras.callbacks.Callback):
    def __init__(self, directory, patience=0):
        super(StopCallback, self).__init__()
        self.stop_flag = False
        self.save_dir = directory

        def time_to_quit(sig, frame):
            self.stop_flag = True
            print('\nStopping at end of this epoch\n')

        signal.signal(signal.SIGINT, time_to_quit)
        self.patience = patience

        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        if self.stop_flag is False:
            self.wait = 0
        else:
            print("\n Keyboard interrupt...")
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print('\n Restoring model weights from the end of the best epoch.')
                self.model.save(os.path.join(self.save_dir, 'model_{:02d}.h5'.format(self.stopped_epoch + 1)))

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch {:02d}: early stopping'.format(self.stopped_epoch + 1))

class CustomTQDMProgressBar(tfa.callbacks.TQDMProgressBar):
    def __init__(self, _custom_total_steps = None):
        super(CustomTQDMProgressBar, self).__init__(metrics_separator= '-',
        epoch_bar_format='{n_fmt}/{total_fmt} ETA:{remaining}s {desc}',
        metrics_format= '{name}:{value:0.4f}',
        show_overall_progress=False)
        self.custom_total_steps = _custom_total_steps

    def _initialize_progbar(self, hook, epoch, logs=None):
        self.num_samples_seen = 0
        self.steps_to_update = 0
        self.steps_so_far = 0
        self.logs = defaultdict(float)
        self.num_epochs = self.params["epochs"]
        self.mode = "steps"
        if self.custom_total_steps is None:
            print("steps: ", self.params["steps"])
            self.total_steps = self.params["steps"]
        else:
            self.total_steps = self.custom_total_steps
        if hook == "train_overall":
            if self.show_overall_progress:
                self.overall_progress_tqdm = self.tqdm(
                    desc="Training",
                    total=self.num_epochs,
                    bar_format=self.overall_bar_format,
                    leave=self.leave_overall_progress,
                    dynamic_ncols=True,
                    unit="epochs",
                )
        elif hook == "test":
            if self.show_epoch_progress:
                self.epoch_progress_tqdm = self.tqdm(
                    total=self.total_steps,
                    desc="Evaluating",
                    bar_format=self.epoch_bar_format,
                    leave=self.leave_epoch_progress,
                    dynamic_ncols=True,
                    unit=self.mode,
                )
        elif hook == "train_epoch":
            current_epoch_description = "Epoch {epoch}/{num_epochs}".format(
                epoch=epoch + 1, num_epochs=self.num_epochs
            )
            if self.show_epoch_progress:
                print(current_epoch_description)
                self.epoch_progress_tqdm = self.tqdm(
                    total=self.total_steps,
                    bar_format=self.epoch_bar_format,
                    leave=self.leave_epoch_progress,
                    dynamic_ncols=True,
                    unit=self.mode,
                )

    def format_metrics(self, logs={}, factor=1):
        """Format metrics in logs into a string.
        Arguments:
            logs: dictionary of metrics and their values. Defaults to
                empty dictionary.
            factor (int): The factor we want to divide the metrics in logs
                by, useful when we are computing the logs after each batch.
                Defaults to 1.
        Returns:
            metrics_string: a string displaying metrics using the given
            formators passed in through the constructor.
        """

        metric_value_pairs = []
        for i,(key, value) in enumerate(logs.items()):
            if key in ["batch", "size"] or any(x in key for x in ["acc", "mse"]):
                continue
            pair = self.metrics_format.format(name=self.choose_name_2(key,i), value=value / factor)
            metric_value_pairs.append(pair)
        metrics_string = self.metrics_separator.join(metric_value_pairs)

        return metrics_string

    def choose_name_2(self, name, index):
        if "loss" in name:
            if "loss" == name:
                return "Ltot"
            else:
                return "L" + str(index)
        elif "mse" in name:
            return "mse" + str(index)
        elif "acc" in name:
            return "acc" + str(index)
        else:
           return str(index) 


    def choose_name(self, name):
        matches = ["one", "minus", "keys"]
        new_name = ""
        name_split = name.split('_')
        for i,s in enumerate(name_split):
            if not(any(x in s for x in matches)):
                if "loss" in s:
                    s = "L"
                elif i < (len(name_split)-1):
                    s += "_"
                new_name+=s
        return new_name

class CustomProgbar(tf.keras.utils.Progbar):

    def __init__(self,
                target,
                width=30,
                verbose=1,
                interval=0.05,
                stateful_metrics=None,
                unit_name='step'):
        super(CustomProgbar, self).__init__(target, width, verbose, interval,\
                                            stateful_metrics, unit_name)

    def update(self, current, values=None, finalize=None):
        """Updates the progress bar.
        Arguments:
            current: Index of current step.
            values: List of tuples: `(name, value_for_last_step)`. If `name` is in
            `stateful_metrics`, `value_for_last_step` will be displayed as-is.
            Else, an average of the metric over time will be displayed.
            finalize: Whether this is the last update for the progress bar. If
            `None`, defaults to `current >= self.target`.
        """
        if finalize is None:
            if self.target is None:
                finalize = False
            else:
                finalize = current >= self.target

        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                value_base = max(current - self._seen_so_far, 1)
                if k not in self._values:
                    self._values[k] = [v * value_base, value_base]
                else:
                    self._values[k][0] += v * value_base
                    self._values[k][1] += value_base
            else:
                self._values[k] = [v, 1]
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if now - self._last_update < self.interval and not finalize:
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.log10(self.target)) + 1
                bar = ('%' + str(numdigits) + 'd/%d [') % (current, self.target)
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

            if self.target is None or finalize:
                if time_per_unit >= 1 or time_per_unit == 0:
                    info += ' %.0fs/%s' % (time_per_unit, self.unit_name)
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/%s' % (time_per_unit * 1e3, self.unit_name)
                else:
                    info += ' %.0fus/%s' % (time_per_unit * 1e6, self.unit_name)
            else:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                    (eta % 3600) // 60, eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if finalize:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if finalize:
                numdigits = int(np.log10(self.target)) + 1
                count = ('%' + str(numdigits) + 'd/%d') % (current, self.target)
                info = count + info
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def choose_name(self, name):
        new_name = ""
        name_split = name.split('_')
        for s in name_split:
            if ["one", "minus", "keys"] not in s:
                if "loss" in s:
                    s = "L"
                elif len(s) > 1:
                    s = s[:2]
                new_name+=s
        return new_name
    

class CustomProgbarLogger(tf.keras.callbacks.ProgbarLogger):
    def __init__(self, verbose, count_mode='samples'):
        super(CustomProgbarLogger, self).__init__(count_mode)
        self.verbose = verbose
        print("self.verbose: ",self.verbose)

    def _maybe_init_progbar(self):
        if self.stateful_metrics is None:
            if self.model:
                self.stateful_metrics = (set(m.name for m in self.model.metrics))
            else:
                self.stateful_metrics = set()

        if self.progbar is None:
            self.progbar = CustomProgbar(
                target=self.target,
                verbose=self.verbose,
                stateful_metrics=self.stateful_metrics,
                unit_name='step' if self.use_steps else 'sample')
