import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Concrete exception logger that logs to the console using java.util.logging.
 */
public class ConsoleExceptionLogger extends AbstractExceptionLogger {
  private static final Logger LOGGER = Logger.getLogger(ConsoleExceptionLogger.class.getName());

  @Override
  protected void actuallyLog(String message, Throwable exception) {
    if (message != null) {
      if (exception != null) {
        LOGGER.log(Level.SEVERE, message, exception);
      } else {
        LOGGER.severe(message);
      }
    } else {
      if (exception != null) {
        LOGGER.log(Level.SEVERE, "Exception occurred", exception);
      } else {
        LOGGER.severe("Exception occurred");
      }
    }
  }
}


