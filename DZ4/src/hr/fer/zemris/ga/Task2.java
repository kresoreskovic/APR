package hr.fer.zemris.ga;

import java.util.Random;

public class Task2 {

	public static void main(String[] args) {
		int popSize = 40;
		double pm = 0.3;
		Random rand = new Random();
		int generationLimit = 1000000;
		int elitism = 1;
		boolean binary = false;
		int p = 5;
		int k = 3;
		int evalLimit = 1000000;
		boolean verbose = false;

		IFunction f = null;
		int[] dim = new int[] { 1, 3, 6 };
		for (int n = 6; n <= 7; n++) {
			System.out.println("\nFunction f" + n + ":");
			for (int i = 0; i < dim.length; i++) {
				int m = dim[i];

				if (n == 6) {
					f = new F6(m, null, new double[m]);
				} else {
					f = new F7(m, null, new double[m]);
				}

				GeneticAlgorithm ga = new GeneticAlgorithm(f, popSize, pm, rand, generationLimit, elitism, binary, p, k,
						f.getDimension(), evalLimit, verbose);
				Chromosome sol1 = ga.generativeGA();
				Chromosome sol2 = ga.eliminativeGA();
				System.out.println(
						"\nGenerative solution for f" + n + ", dimension " + m + ": " + sol1 + ", " + sol1.getFitness());
				System.out.println(
						"Eliminative solution for f" + n + ", dimension " + m + ": " + sol2 + ", " + sol2.getFitness());
				System.out.println(
						"---------------------------------------------------------------------------------------");
			}
		}
	}

}
