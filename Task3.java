public class Task3 {
    static final long ITERS = 50_000_000L;
    static final int PAD = 16;
    static int THREADS = 8;

    interface Work { void run(int i, long[] arr, long n); }

    static void unpadded(int i, long[] a, long n) {
        for (long k = 0; k < n; k++) a[i]++;
    }

    static void padded(int i, long[] a, long n) {
        int idx = i * PAD;
        for (long k = 0; k < n; k++) a[idx]++;
    }

    static void local(int i, long[] a, long n) {
        long c = 0;
        for (long k = 0; k < n; k++) c++;
        a[i] = c;
    }

    static double run(Work w, long[] arr, long n) throws InterruptedException {
        Thread[] ts = new Thread[THREADS];
        for (int t = 0; t < THREADS; t++) {
            final int id = t;
            ts[t] = new Thread(() -> w.run(id, arr, n));
        }
        long t0 = System.nanoTime();
        for (Thread t : ts) t.start();
        for (Thread t : ts) t.join();
        return (System.nanoTime() - t0) / 1e6;
    }

    public static void main(String[] args) throws Exception {
        if (args.length > 0) THREADS = Integer.parseInt(args[0]);

        System.out.println("Student ID     : 230103240");
        System.out.println("OS             : " + System.getProperty("os.name") + " " + System.getProperty("os.version"));
        System.out.println("Arch           : " + System.getProperty("os.arch"));
        System.out.println("Logical cores  : " + Runtime.getRuntime().availableProcessors());
        System.out.println("Worker threads : " + THREADS);
        System.out.println("Increments/thr : " + String.format("%,d", ITERS) + "\n");

        long[] w = new long[THREADS * PAD];
        run(Task3::unpadded, w, 2_000_000L);
        run(Task3::padded, w, 2_000_000L);
        run(Task3::local, w, 2_000_000L);

        long[] a1 = new long[THREADS];
        long[] a2 = new long[THREADS * PAD];
        long[] a3 = new long[THREADS];

        double tUnpadded = run(Task3::unpadded, a1, ITERS);
        double tPadded   = run(Task3::padded,   a2, ITERS);
        double tLocal    = run(Task3::local,    a3, ITERS);

        for (int i = 0; i < THREADS; i++) {
            if (a1[i] != ITERS || a2[i * PAD] != ITERS || a3[i] != ITERS)
                throw new IllegalStateException("count mismatch at thread " + i);
        }

        System.out.printf("Unpadded shared : %10.2f ms%n", tUnpadded);
        System.out.printf("Padded (128 B)  : %10.2f ms%n", tPadded);
        System.out.printf("Local variable  : %10.2f ms%n", tLocal);
        System.out.printf("%nRatio Unpadded / Padded : %.2fx slower%n", tUnpadded / tPadded);
        System.out.printf("Ratio Unpadded / Local  : %.2fx slower%n", tUnpadded / tLocal);
    }
}