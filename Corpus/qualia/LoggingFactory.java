/**
 * Lightweight factory for default logging implementations.
 */
public final class LoggingFactory {
  private LoggingFactory() {}

  public static ExceptionLogger createLogger() {
    // Default to console-based logging
    return new ConsoleExceptionLogger();
  }
}
