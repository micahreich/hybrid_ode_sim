from enum import Enum


class LogLevel(Enum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


FORMATS = {
    LogLevel.DEBUG: "\033[0mDEBUG:\033[0m",  # white
    LogLevel.INFO: "\033[32mINFO:\033[0m",  # green
    LogLevel.WARNING: "\033[33mWARNING:\033[0m",  # yellow
    LogLevel.ERROR: "\033[31mERROR:\033[0m",  # red
    LogLevel.CRITICAL: "\033[41mCRITICAL:\033[0m",  # red background
}


class Logger:
    def __init__(self, level=LogLevel.DEBUG, name=None) -> None:
        """
        Initialize the LoggingTool object.

        Parameters
        ----------
        level : LogLevel, optional
            The log level to be set for the LoggingTool object. The default value is LogLevel.DEBUG.
        name : str, optional
            The name of the LoggingTool object. The default value is None.
        """
        self.level = level
        self.name = name

    def _log(self, level: LogLevel, message: str):
        log_string = f"{FORMATS.get(level)} {message}"

        if self.name:
            log_string = f"[{self.name}] {log_string}"

        print(log_string)

    def debug(self, message: str):
        if self.level.value <= LogLevel.DEBUG.value:
            self._log(LogLevel.DEBUG, message)

    def info(self, message: str):
        if self.level.value <= LogLevel.INFO.value:
            self._log(LogLevel.INFO, message)

    def warning(self, message: str):
        if self.level.value <= LogLevel.WARNING.value:
            self._log(LogLevel.WARNING, message)

    def error(self, message: str):
        if self.level.value <= LogLevel.ERROR.value:
            self._log(LogLevel.ERROR, message)

    def critical(self, message: str):
        if self.level.value <= LogLevel.CRITICAL.value:
            self._log(LogLevel.CRITICAL, message)


if __name__ == "__main__":
    logger = Logger(LogLevel.DEBUG, "TestLogger")

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
