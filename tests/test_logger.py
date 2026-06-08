import pytest
from BLLMC.utils.logger import SingletonLogger, logger


class TestSingletonLogger:
    """Unit tests for SingletonLogger."""

    def test_singleton_identity(self):
        """Test that multiple instantiations refer to the exact same object."""
        logger1 = SingletonLogger()
        logger2 = SingletonLogger()
        assert logger1 is logger2

    def test_module_logger_is_singleton(self):
        """Test that the imported logger is the singleton instance."""
        assert isinstance(logger, SingletonLogger)

    def test_logger_methods(self):
        """Test that logger delegates methods correctly without raising errors."""
        try:
            logger.info("Test INFO log message")
            logger.warning("Test WARNING log message")
            logger.debug("Test DEBUG log message")
        except AttributeError as e:
            pytest.fail(f"Logger delegation failed with AttributeError: {e}")
