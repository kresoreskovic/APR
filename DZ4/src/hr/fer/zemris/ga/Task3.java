package hr.fer.zemris.ga;

import java.util.Random;

public class Task3 {

	public static void main(String[] args) {
		int popSize = 40;
		double pm = 0.1;
		Random rand = new Random();
		int generationLimit = 1000000;
		int elitism = 1;
		boolean binary = true;
		int p = 4;
		int k = 3;
		int evalLimit = 1000000;
		boolean verbose = false;

		int[] sizes = new int[] { 3, 6 };
		for (int n = 0; n <= 1; n++) {
			int m = sizes[n];
			for (int i = 0; i < 10; i++) {
				IFunction f = new F7(m, null, new double[m]);
				GeneticAlgorithm ga = new GeneticAlgorithm(f, popSize, pm, rand, generationLimit, elitism, binary, p, k,
						f.getDimension(), evalLimit, verbose);
				Chromosome sol = ga.eliminativeGA();
				System.out.println("Eliminative binary solution for f7, dimension " + m + ": " + sol + ", " + sol.getFitness());
			}
			System.out.println();
			for (int i = 0; i < 10; i++) {
				IFunction f = new F7(m, null, new double[m]);
				GeneticAlgorithm ga = new GeneticAlgorithm(f, popSize, pm, rand, generationLimit, elitism, false, p, k,
						f.getDimension(), evalLimit, verbose);
				Chromosome sol = ga.eliminativeGA();
				System.out.println("Eliminative double solution for f7, dimension " + m + " : " + sol + ", " + sol.getFitness());
			}
			System.out.println();
			System.out.println();
		}

	}

}
