package hr.fer.zemris.ga;

import java.util.Random;

public class Task5 {

	public static void main(String[] args) {
		int popSize = 40;
		double pm = 0.1;
		Random rand = new Random();
		int generationLimit = 1000000;
		int elitism = 1;
		boolean binary = false;
		int p = 4;
		int evalLimit = 1000000;
		boolean verbose = false;

		System.out.println("2,3,4,5,6");
		int[] ks = new int[] { 2, 3, 4, 5, 6 };
		for (int n = 0; n < 10; n++) {
			for (int i = 0; i < ks.length; i++) {
				int k = ks[i];
				IFunction f6 = new F6(6, null, new double[6]);
				GeneticAlgorithm ga = new GeneticAlgorithm(f6, popSize, pm, rand, generationLimit, elitism, binary, p,
						k, f6.getDimension(), evalLimit, verbose);
				Chromosome sol = ga.eliminativeGA();
				System.out.print(sol.getFitness());
				if (i < ks.length - 1) {
					System.out.print(",");
				}
			}
			System.out.println();
		}
	}

}
