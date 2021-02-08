import argparse
import datetime
import logging
import math
import re
import subprocess
import sys

from GPyOpt.methods import BayesianOptimization
import yaml

DEFAULT_LOSS_VALUE = 1.


class MLBlackBox():

  def __init__(self, model_config, domain, dataset):
    self.model_config = model_config
    self.domain = domain
    self.dataset = dataset
    self.count = 0

  @staticmethod
  def get_loss(logfile):
    losses = []
    epoch = -1
    epoch_pattern = r'Epoch (\d+)'
    loss_pattern = r'val_loss: (\S+)'
    with open(logfile) as f:
      for line in f:
        line = line.rstrip()
        epoch_match = re.search(epoch_pattern, line)
        if epoch_match:
          epoch = int(epoch_match.group(1))
        loss_match = re.search(loss_pattern, line)
        if loss_match:
          loss = float(loss_match.group(1))
          losses.append((loss, epoch))
    if len(losses) > 0:
      best_loss, best_epoch = min(losses)
      logging.info('Best evaluation loss found at Epoch %s: %s',
                   best_epoch, best_loss)
      logging.info(
          '----------------------------------------------------------')
    else:
      best_loss = DEFAULT_LOSS_VALUE
    return best_loss

  def run_model(self, model_config):
    cmd = 'bin/train.sh {} {} {}'.format(model_config,
                                         self.dataset, 'hptuning')
    logging.info(cmd)
    output = subprocess.run(cmd, stdout=subprocess.PIPE,
                            shell=True, check=False)
    output = output.stdout.decode('utf-8')
    output = output.splitlines()[-1]
    logging.info(output)
    return re.match(r'Logging to file: (\S+)', output).group(1)

  def _create_model_config(self, hparams):
    lines = open('config/{}.sh'.format(self.model_config), 'r').readlines()
    last_line = lines.pop()
    for i, scaled_value in enumerate(hparams):
      value = rescale_value(scaled_value, self.domain[i]['scale'])
      lines.append('  --{}={} \\\n'.format(self.domain[i]['name'], value))
    lines.append(last_line)
    now = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())
    model_config = '{}_{}_{:03d}'.format(now, self.model_config, self.count)
    script = 'config/autogenerated/{}.sh'.format(model_config)
    open(script, 'w').writelines(lines)
    return model_config

  def f(self, hparams):
    model_config = self._create_model_config(hparams[0])
    self.count += 1
    log = self.run_model(model_config)
    return self.get_loss(log)


def rescale_value(value, scale):
  if scale == 'LINEAR_SCALE':
    return int(value)
  if scale == 'DECIMAL_SCALE':
    return value / 10
  if scale == 'LOG2_SCALE':
    return int(2**value)
  if scale == 'LOG10_SCALE':
    return int(10**value)
  raise ValueError(
      'Unsupported scale. Expected LINEAR_SCALE, DECIMAL_SCALE, LOG2_SCALE,'
      ' or LOG10_SCALE. Got {} instead.'.format(scale))


def get_scaled_values(min_value, max_value, scale):
  if scale == 'LINEAR_SCALE':
    return list(range(int(min_value), int(max_value + 1)))
  if scale == 'DECIMAL_SCALE':
    return list(range(int(min_value * 10), int(max_value * 10 + 1)))
  if scale == 'LOG2_SCALE':
    return list(range(int(math.log2(min_value)), int(math.log2(max_value) + 1)))
  if scale == 'LOG10_SCALE':
    return list(range(int(math.log(min_value)), int(math.log(max_value) + 1)))
  raise ValueError(
      'Unsupported scale. Expected LINEAR_SCALE, DECIMAL_SCALE, LOG2_SCALE,'
      ' or LOG10_SCALE. Got {} instead.'.format(scale))


def get_hparams(hptuning_config):
  with open(hptuning_config) as f:
    hparams = f.read()
  return yaml.load(hparams)


def get_domain(hptuning_config):
  hparams = get_hparams(hptuning_config)

  domain = []
  for hparam in hparams['hyperparameters']:
    values = get_scaled_values(
        float(hparam['min_value']), float(hparam['max_value']), hparam['scale'])
    domain.append({'name': hparam['name'], 'domain': values,
                   'type': 'discrete', 'scale': hparam['scale']})
  return domain, hparams['max_trials']


def run_experiment(model_config, hptuning_config, dataset, label):
  domain, max_trials = get_domain(hptuning_config)
  black_box = MLBlackBox(model_config, domain, dataset)
  optimizer = BayesianOptimization(
      f=black_box.f, domain=domain, normalize_Y=False)
  report_file = 'log/hptuning_report_{}.log'.format(label)
  logging.info('Logging to %s', report_file)
  optimizer.run_optimization(
      verbosity=True,
      max_iter=max_iterations,
      report_file=report_file,
      evaluations_file='log/hptuning_evaluations_{}.log'.format(label),
      models_file='log/hptuning_model_{}.log'.format(label),
  )


def _set_logging(log_level):
  logging.basicConfig(level=log_level)


def _parse_arguments(argv):
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model_config',
      help='ML model configuration.',
      required=True)
  parser.add_argument(
      '--hptuning_config',
      help='Hyperparameter tuning configuration.',
      required=True)
  parser.add_argument(
      '--dataset',
      help='Dataset identifier.',
      required=True)
  parser.add_argument(
      '--label',
      help='Label.',
      required=True)
  parser.add_argument(
      '--log_level',
      help='Set log level.',
      default='INFO')
  return parser.parse_args(argv)


def main():
  args = _parse_arguments(sys.argv[1:])
  _set_logging(args.log_level.upper())
  run_experiment(args.model_config, args.hptuning_config,
                 args.dataset, args.label)


if __name__ == '__main__':
  main()
