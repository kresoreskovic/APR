package hr.fer.zemris.ga;

import java.util.Random;

public class Task4 {

	public static void main(String[] args) {
		int popSize = 40;
		double pm = 0.1;
		Random rand = new Random();
		int generationLimit = 1000000;
		int elitism = 1;
		boolean binary = false;
		int p = 4;
		int k = 3;
		int evalLimit = 1000000;
		boolean verbose = false;

		System.out.println("30,50,100,200");
		int[] popSizes = new int[] { 30, 50, 100, 200 };
		for (int n = 1; n <= 10; n++) {
			for (int i = 0; i < popSizes.length; i++) {
				int popSizeTmp = popSizes[i];
				IFunction f6 = new F6(6, null, new double[6]);
				GeneticAlgorithm ga6 = new GeneticAlgorithm(f6, popSizeTmp, pm, rand, generationLimit, elitism, binary,
						p, k, f6.getDimension(), evalLimit, verbose);
				Chromosome sol = ga6.generativeGA();
				
				System.out.print(sol.getFitness());
				if (i < popSizes.length - 1) {
					System.out.print(",");
				}
			}
			System.out.println();
		}

		System.out.println();
		System.out.println("0.1,0.3,0.6,0.9");
		double[] pmSizes = new double[] { 0.1, 0.3, 0.6, 0.9 };
		for (int n = 1; n <= 10; n++) {
			for (int i = 0; i < popSizes.length; i++) {
				double pmTmp = pmSizes[i];
				IFunction f6 = new F6(6, null, new double[6]);
				GeneticAlgorithm ga6 = new GeneticAlgorithm(f6, popSize, pmTmp, rand, generationLimit, elitism, binary,
						p, k, f6.getDimension(), evalLimit, verbose);
				Chromosome sol = ga6.generativeGA();
				
				System.out.print(sol.getFitness());
				if (i < popSizes.length - 1) {
					System.out.print(",");
				}
			}
			System.out.println();
		}
	}

}
