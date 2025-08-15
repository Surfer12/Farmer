import java.util.Objects;

/**
 * Abstract base class for exception loggers.
 * Provides a simple delegation surface to concrete implementations.
 */
public abstract class AbstractExceptionLogger implements ExceptionLogger {
  /**
   * Subclasses implement the actual logging mechanism.
   * @param message Optional message to accompany the exception
   * @param exception The exception to log
   */
  protected abstract void actuallyLog(String message, Throwable exception);

  @Override
  public void log(Exception exception) {
    actuallyLog(null, exception);
  }

  @Override
  public void log(String message, Exception exception) {
    Objects.requireNonNull(exception, "exception must not be null");
    actuallyLog(message, exception);
  }
}


