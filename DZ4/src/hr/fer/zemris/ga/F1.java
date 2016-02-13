package hr.fer.zemris.ga;

public class F1 implements IFunction {

	public int N;
	public double[] start;
	public double[] minPoint;
	public int eval;

	public F1(int N, double[] start, double[] minPoint) {
		this.N = N;
		this.start = start;
		this.minPoint = minPoint;
	}

	@Override
	public double valueAt(double[] point) {
		this.eval++;
		return 100 * Math.pow((point[1] - Math.pow(point[0], 2)), 2) + Math.pow(1 - point[0], 2);
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
