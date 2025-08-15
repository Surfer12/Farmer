import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * Minimal compatibility wrapper around ThreadPoolExecutor.
 * This exists to satisfy legacy code that references
 * {@code ThreadPoolExecutorCompat} without pulling in a full
 * compatibility layer. It simply delegates to the standard
 * ThreadPoolExecutor implementation.
 */
public class ThreadPoolExecutorCompat extends ThreadPoolExecutor {
  public ThreadPoolExecutorCompat(int corePoolSize,
                                int maximumPoolSize,
                                long keepAliveTime,
                                TimeUnit unit,
                                BlockingQueue<Runnable> workQueue,
                                ThreadFactory threadFactory) {
    super(corePoolSize, maximumPoolSize, keepAliveTime, unit, workQueue, threadFactory);
  }
}


