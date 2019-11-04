import logging
import click


class ClickColoredLoggingFormatter(logging.Formatter):
    default_colors = {
        "error": dict(fg="red"),
        "exception": dict(fg="red"),
        "critical": dict(fg="red"),
        "debug": dict(fg="blue"),
        "warning": dict(fg="yellow"),
    }

    def __init__(self, fmt=None, datefmt=None, colors=None):
        super(ClickColoredLoggingFormatter, self).__init__(fmt, datefmt)
        self.colors = colors or ClickColoredLoggingFormatter.default_colors

    def format(self, record):
        if not record.exc_info:
            level = record.levelname.lower()
            msg = record.getMessage()
            if level in self.colors:
                prefix = click.style("{}: ".format(level), **self.colors[level])
                # msg = "\n".join(prefix + x for x in msg.splitlines())
                msg = prefix + logging.Formatter.format(self, record)
            return msg
        return logging.Formatter.format(self, record)


class ClickLoggingHandler(logging.Handler):
    _use_stderr = True

    def emit(self, record):
        try:
            msg = self.format(record)
            click.echo(msg, err=self._use_stderr)
        except Exception:
            self.handleError(record)
