import java.util.Locale;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

public class Task6 {
    static final int THREADS = 4;
    static final int PER_THREAD = 2_000_000;
    static final int EXPECTED = THREADS * PER_THREAD;

    static volatile int unsafeCounter;
    static final AtomicInteger atomicCounter = new AtomicInteger();
    static final ReentrantLock lock = new ReentrantLock();
    static int lockedCounter;

    static double run(Runnable body) throws InterruptedException {
        Thread[] ts = new Thread[THREADS];
        for (int i = 0; i < THREADS; i++) ts[i] = new Thread(body);
        long t0 = System.nanoTime();
        for (Thread t : ts) t.start();
        for (Thread t : ts) t.join();
        return (System.nanoTime() - t0) / 1e6;
    }

    static void unsafeWork() {
        for (int k = 0; k < PER_THREAD; k++) unsafeCounter++;
    }

    static void atomicWork() {
        for (int k = 0; k < PER_THREAD; k++) atomicCounter.incrementAndGet();
    }

    static void lockWork() {
        for (int k = 0; k < PER_THREAD; k++) {
            lock.lock();
            try {
                lockedCounter++;
            } finally {
                lock.unlock();
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.US);

        System.out.println("Student ID     : 230103240");
        System.out.println("OS             : " + System.getProperty("os.name") + " " + System.getProperty("os.version"));
        System.out.println("Logical cores  : " + Runtime.getRuntime().availableProcessors());
        System.out.printf("Threads        : %d x %,d increments%n%n", THREADS, PER_THREAD);

        run(Task6::unsafeWork);
        run(Task6::atomicWork);
        run(Task6::lockWork);
        unsafeCounter = 0;
        atomicCounter.set(0);
        lockedCounter = 0;

        double tUnsafe = run(Task6::unsafeWork);
        double tAtomic = run(Task6::atomicWork);
        double tLock = run(Task6::lockWork);

        int actual = unsafeCounter;
        long lost = (long) EXPECTED - actual;
        double lostPct = 100.0 * lost / EXPECTED;

        System.out.println("--- Unsynchronized (race condition) ---");
        System.out.printf("Expected final value : %,d%n", EXPECTED);
        System.out.printf("Actual final value   : %,d%n", actual);
        System.out.printf("Updates lost         : %,d (%.2f%%)%n", lost, lostPct);
        System.out.printf("Wall time            : %.2f ms%n%n", tUnsafe);

        System.out.println("--- AtomicInteger ---");
        System.out.printf("Final value          : %,d (%s)%n", atomicCounter.get(),
                atomicCounter.get() == EXPECTED ? "correct" : "WRONG");
        System.out.printf("Wall time            : %.2f ms%n%n", tAtomic);

        System.out.println("--- ReentrantLock ---");
        System.out.printf("Final value          : %,d (%s)%n", lockedCounter,
                lockedCounter == EXPECTED ? "correct" : "WRONG");
        System.out.printf("Wall time            : %.2f ms%n%n", tLock);

        System.out.println("--- Synchronization penalty vs unsynchronized ---");
        System.out.printf("AtomicInteger        : %+.2f ms (%.2fx)%n", tAtomic - tUnsafe, tAtomic / tUnsafe);
        System.out.printf("ReentrantLock        : %+.2f ms (%.2fx)%n", tLock - tUnsafe, tLock / tUnsafe);
    }
}