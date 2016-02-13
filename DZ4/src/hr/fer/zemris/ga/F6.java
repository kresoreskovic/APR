package hr.fer.zemris.ga;

public class F6 implements IFunction {

	public int N;
	public double[] start;
	public double[] minPoint;
	public int eval;

	public F6(int N, double[] start, double[] minPoint) {
		this.N = N;
		this.start = start;
		this.minPoint = minPoint;
	}

	@Override
	public double valueAt(double[] point) {
		this.eval++;
		double sum = 0;
		for (int i = 0; i < N; i++) {
			sum += Math.pow(point[i], 2);
		}
		return 0.5 + (Math.pow(Math.sin(Math.sqrt(sum)), 2) - 0.5) / (Math.pow((1 + 0.001 * sum), 2));
	}

	@Override
	public double fitness(double[] point) {
		return this.valueAt(point);
	}

	@Override
	public void reset() {
		this.eval = 0;
	}

	@Override
	public int getDimension() {
		return this.N;
	}

	@Override
	public double[] getStart() {
		return this.start;
	}

	@Override
	public int getEval() {
		return this.eval;
	}

}
