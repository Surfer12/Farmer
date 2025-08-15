/**
 * Logging interface for exception events.
 * This interface abstracts the ability to log exceptions with an
 * optional contextual message.
 */
public interface ExceptionLogger {
  void log(Exception exception);
  void log(String message, Exception exception);
}


