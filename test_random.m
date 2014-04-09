stream = RandStream('mt19937ar','Seed',sum(100*clock));
RandStream.setDefaultStream(stream);
a = randperm(10);
a
quit;

