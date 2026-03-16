"""MkDocs build hooks for quent documentation."""

from importlib.metadata import version


def on_config(config):
  """Inject the package version into config.extra so templates can use it."""
  config.extra['version'] = version('quent')
  return config
