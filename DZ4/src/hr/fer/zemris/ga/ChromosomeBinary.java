package hr.fer.zemris.ga;

import java.util.Random;

public class ChromosomeBinary implements Comparable<ChromosomeBinary>, Chromosome {

	private double fitness;
	private int[] points;
	private int N;
	private int n;

	public ChromosomeBinary(Random rand, int N, int p) {
		this.fitness = 0;
		this.N = N;
		this.n = (int) (Math.log(Math.floor(1 + (Decoder.hi - Decoder.lo) * Math.pow(10, p))) / Math.log(2));
		this.points = new int[N];
		for (int i = 0; i < N; i++) {
			this.points[i] = rand.nextInt((int) Math.pow(2, n - 1));
		}
	}

	@Override
	public int compareTo(ChromosomeBinary o) {
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
		for (int b : this.points) {
			sb.append(Decoder.convertFromBinary(b, n) + " ");
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
		this.points[index] = Decoder.convertToBinary(value, index);
	}

	@Override
	public double[] getPoints() {
		double[] actualPoints = new double[N];
		for (int i = 0; i < this.points.length; i++) {
			actualPoints[i] = Decoder.convertFromBinary(this.points[i], n);
		}
		return actualPoints;
	}

	@Override
	public double getValue(int index) {
		return Decoder.convertFromBinary(this.points[index], n);
	}

	@Override
	public int[] getBinary(int index) {
		int[] sol = new int[n];
		String binary = Integer.toBinaryString(points[index]);
		int diff = n - binary.length();
		if (diff > 0) {
			for (int i = 0; i < diff; i++) {
				binary = "0" + binary;
			}
		}
		if (n != binary.length()) {
			System.out.println(n + " " + binary.length());
			System.out.println(points[index] + " " + binary);
		}
		for (int j = 0; j < binary.length(); j++) {
			int tmp = Integer.parseInt("" + binary.charAt(j));
			sol[j] = tmp;
		}
		return sol;
	}

}
