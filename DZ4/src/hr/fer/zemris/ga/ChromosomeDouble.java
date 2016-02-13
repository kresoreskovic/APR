package hr.fer.zemris.ga;

import java.util.Random;

public class ChromosomeDouble implements Comparable<ChromosomeDouble>, Chromosome {

	private double fitness;
	private double points[];

	public ChromosomeDouble(Random rand, int N) {
		this.fitness = 0;
		this.points = new double[N];
		if (rand != null) {
			for (int i = 0; i < N; i++) {
				this.points[i] = rand.nextDouble() * (Decoder.hi - Decoder.lo) + Decoder.lo;
			}
		}
	}

	@Override
	public int compareTo(ChromosomeDouble o) {
		if (this.fitness < o.fitness) {
			return -1;
		}
		if (this.fitness > o.fitness) {
			return 1;
		}
		return 0;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder("[ ");
		for (double d : this.points) {
			sb.append(d + " ");
		}
		sb.append("]");
		return sb.toString();
	}

	@Override
	public double getFitness() {
		return this.fitness;
	}

	@Override
	public void setFitness(double fitness) {
		this.fitness = fitness;
	}

	@Override
	public int getSize() {
		return this.points.length;
	}

	@Override
	public void setValue(int index, double value) {
		this.points[index] = value;
	}

	@Override
	public double[] getPoints() {
		return this.points;
	}

	@Override
	public double getValue(int index) {
		return this.points[index];
	}

	@Override
	public int[] getBinary(int index) {
		return null;
	}

}
