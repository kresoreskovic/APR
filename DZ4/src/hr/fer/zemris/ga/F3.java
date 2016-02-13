package hr.fer.zemris.ga;

public class F3 implements IFunction {

	public int N;
	public double[] start;
	public double[] minPoint;
	public int eval;

	public F3(int N, double[] start, double[] minPoint) {
		this.N = N;
		this.start = start;
		this.minPoint = minPoint;
	}

	@Override
	public double valueAt(double[] point) {
		this.eval++;
		double sol = 0;
		for (int i = 0; i < N; i++) {
			sol += Math.pow((point[i] - i), 2);
		}
		return sol;
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
