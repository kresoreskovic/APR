package hr.fer.zemris.ga;

public class F7 implements IFunction {

	public int N;
	public double[] start;
	public double[] minPoint;
	public int eval;

	public F7(int N, double[] start, double[] minPoint) {
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
		return Math.pow(sum, 0.25) * (1 + Math.pow(Math.sin(50 * (Math.pow(sum, 0.1))), 2));
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
