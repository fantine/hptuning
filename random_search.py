import datetime
import pickle
import random
import numpy as np
import os
import re
import subprocess
import sys
import argparse
import logging

from GPyOpt.methods import BayesianOptimization
import yaml
import matplotlib
matplotlib.use('Agg')


np.random.seed(42)


class MLBlackBox(object):

  def __init__(self, model_template, domain, dataset):
    self.model_template = model_template
    self.template_script = os.path.join(
        'config', '{}.sh'.format(model_template))
    self.domain = domain
    self.dataset = dataset

  def _run_ml_model(self, model):
    cmd = 'bin/train_fg.sh {} {}'.format(model, self.dataset)
    logging.info(cmd)
    output = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
    output = output.stdout.decode('utf-8')
    output = output.splitlines()[-1]
    logging.info(output)
    return re.match('Logging to file: (\S+)', output).group(1)

  def _get_loss(self, logfile):
    eval_losses = []
    logs = []
    epoch = -1
    epoch_pattern = 'Epoch (\d+)'
    loss_pattern = 'val_loss: (\S+)'
    with open(logfile) as f:
      for line in f:
        line = line.rstrip()
        current_epoch = re.search(epoch_pattern, line)
        if current_epoch:
          logging.info(line)
          epoch = int(current_epoch.group(1))
        loss = re.search(loss_pattern, line)
        if loss:
          eval_losses.append(float(loss.group(1)))
          logging.info(line)
          logs.append((float(loss.group(1)), epoch))
    if len(eval_losses) > 1:
      smoothed_loss = []
      for i in range(1, len(eval_losses)):
        smoothed_loss.append((eval_losses[i] + eval_losses[i - 1]) / 2)
      min_index = np.argmin(smoothed_loss)
      minimum_loss = smoothed_loss[min_index]
      _, best_epoch = logs[min_index + 1]
      logging.info(
          'Best evaluation loss found at Epoch {}: val_loss: {}'.format(best_epoch, minimum_loss))
      logging.info(
          '----------------------------------------------------------')
    else:
      minimum_loss = 1e15
    return minimum_loss

  def _create_script(self, hparams):
    values_str = '_'.join([str(val) for val in hparams])
    model = 'hp_{}_{}'.format(self.model_template, values_str)
    lines = open(self.template_script, 'r').readlines()
    last_line = lines.pop()
    for i, val in enumerate(hparams):
      if self.domain[i]['dtype'] == 'INTEGER':
        val = int(val)
      lines.append('  --{}={} \\\n'.format(self.domain[i]['name'], val))
    lines.append(last_line)
    now = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())
    script = os.path.join('config', '{}_{}.sh'.format(model, now))
    open(script, 'w').writelines(lines)
    return '{}_{}'.format(model, now)

  def f(self, hparams):
    hparams = hparams[0]
    model = self._create_script(hparams)
    logfile = self._run_ml_model(model)
    return self._get_loss(logfile)


def load_hparams_config(hparams_config):
  with open(hparams_config) as f:
    hparams_str = f.read()
  hparams = yaml.load(hparams_str)
  return hparams


def get_param_domain(hparams_config):
  hparams = load_hparams_config(hparams_config)
  max_iter = hparams['trainingInput']['hyperparameters']['maxTrials']

  domain = []
  for hp in hparams['trainingInput']['hyperparameters']['params']:
    name = hp['parameterName']
    minval = float(hp['minValue'])
    maxval = float(hp['maxValue'])
    scaletype = hp['scaleType']
    dtype = hp['type']

    if scaletype == 'UNIT_LINEAR_SCALE':
      values = list(range(int(minval), int(maxval + 1)))
    elif scaletype == 'UNIT_LOG_SCALE':
      values = [minval]
      while values[-1] < maxval:
        values.append(values[-1] * 10)
    elif scaletype == 'UNIT_LOG2_SCALE':
      values = [int(minval)]
      while values[-1] < maxval:
        values.append(int(values[-1] * 2))
    else:
      logging.error("Parameter {} has an unknown scale type".format(name))

    domain.append({'name': name, 'type': 'discrete', 'domain': values,
                   'dtype': dtype})

  return domain, max_iter


def run_experiment(model_template, hparams_config, dataset):
  now = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())
  logging.info('Logging to evaluations_{}.txt'.format(now))
  domain, max_iter = get_param_domain(hparams_config)

  report_file = 'evaluations_{}.txt'.format(now)
  var_names = []
  for sub_domain in domain:
    var_names.append(sub_domain['name'])
  var_names = '\t'.join(var_names)
  with open(report_file, 'w') as f:
    f.write('iteration\teval_loss\t{}\n'.format(var_names))
  black_box = MLBlackBox(model_template, domain, dataset)
  for i in range(int(max_iter)):
    hparams = [[]]
    # domain is a list of dictionaries
    for sub_domain in domain:
      hparams[0].append(random.choice(sub_domain['domain']))
    eval_loss = black_box.f(hparams)
    with open(report_file, 'a') as f:
      hparam_str = '\t'.join([str(hp) for hp in hparams[0]])
      f.write('{}\t{}\t{}\n'.format(i + 1, eval_loss, hparam_str))

  # optimizer = BayesianOptimization(
  #     f=black_box.f,
  #     domain=domain,
  #     normalize_Y=False
  # )
  # optimizer.run_optimization(verbosity=True, max_iter=max_iter,
  #                            report_file='report_{}.txt'.format(now),
  #                            evaluations_file='evaluations_{}.txt'.format(now),
  #                            models_file='models_{}.txt'.format(now))

  # optimizer.plot_acquisition(filename='plot_acquisition_{}.png'.format(now))
  # optimizer.plot_convergence(filename='plot_convergence_{}.png'.format(now))

  # scratch_path = '/scr1/fantine/earthquake-detection/results/'
  # ckpt_file = os.path.join(scratch_path, 'optimizer_{}.pkl'.format(now))
  # pickle_file = open(ckpt_file, 'wb')
  # pickle.dump(optimizer, pickle_file)


def _set_logging(log_level):
  logging.basicConfig(level=log_level)


def _parse_arguments(argv):
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--hparams_config',
      help='Hyper-parameters configuration file.',
      required=True)
  parser.add_argument(
      '--model_template',
      help='ML model script template.',
      required=True)
  parser.add_argument(
      '--dataset',
      help='Dataset name.',
      required=True)
  parser.add_argument(
      '--log_level',
      help='Set log level.',
      default='INFO')
  return parser.parse_args(argv)


def main():
  args = _parse_arguments(sys.argv[1:])
  _set_logging(args.log_level.upper())
  run_experiment(args.model_template, args.hparams_config, args.dataset)


if __name__ == '__main__':
  main()
