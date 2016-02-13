package hr.fer.zemris.ga;

import java.util.Random;

public class Task1 {

	public static void main(String[] args) {
		int popSize = 40;
		double pm = 0.3;
		Random rand = new Random();
		int generationLimit = 10000000;
		int elitism = 1;
		boolean binary = false;
		int p = 5;
		int k = 3;
		int evalLimit = 10000000;
		boolean verbose = false;

		IFunction f1 = new F1(2, new double[] { -1.9, 2 }, new double[] { 1, 1 });
		IFunction f3 = new F3(5, new double[] { 0, 0, 0, 0, 0 }, new double[] { 1, 2, 3, 4, 5 });
		IFunction f6 = new F6(2, null, new double[] { 0, 0 });
		IFunction f7 = new F7(2, null, new double[] { 0, 0 });

		GeneticAlgorithm ga1 = new GeneticAlgorithm(f1, popSize, pm, rand, generationLimit, elitism, binary, p, k,
				f1.getDimension(), evalLimit, verbose);
		Chromosome sol11 = ga1.generativeGA();
		Chromosome sol12 = ga1.eliminativeGA();
		System.out.println("\nGenerative solution for f1: " + sol11 + ", " + sol11.getFitness());
		System.out.println("Eliminative solution for f1: " + sol12 + ", " + sol12.getFitness());
		System.out.println("---------------------------------------------------------------------------------------");
		
		GeneticAlgorithm ga3 = new GeneticAlgorithm(f3, popSize, pm, rand, generationLimit, elitism, binary, p, k,
				f3.getDimension(), evalLimit, verbose);
		Chromosome sol31 = ga3.generativeGA();
		Chromosome sol32 = ga3.eliminativeGA();
		System.out.println("\nGenerative solution for f3: " + sol31 + ", " + sol31.getFitness());
		System.out.println("Eliminative solution for f3: " + sol32 + ", " + sol32.getFitness());
		System.out.println("---------------------------------------------------------------------------------------");

		GeneticAlgorithm ga6 = new GeneticAlgorithm(f6, popSize, pm, rand, generationLimit, elitism, binary, p, k,
				f6.getDimension(), evalLimit, verbose);
		Chromosome sol61 = ga6.generativeGA();
		Chromosome sol62 = ga6.eliminativeGA();
		System.out.println("\nGenerative solution for f6: " + sol61 + ", " + sol61.getFitness());
		System.out.println("Eliminative solution for f6: " + sol62 + ", " + sol62.getFitness());
		System.out.println("---------------------------------------------------------------------------------------");

		GeneticAlgorithm ga7 = new GeneticAlgorithm(f7, popSize, pm, rand, generationLimit, elitism, binary, p, k,
				f7.getDimension(), evalLimit, verbose);
		Chromosome sol71 = ga7.generativeGA();
		Chromosome sol72 = ga7.eliminativeGA();
		System.out.println("\nGenerative solution for f7: " + sol71 + ", " + sol71.getFitness());
		System.out.println("Eliminative solution for f7: " + sol72 + ", " + sol72.getFitness());
		System.out.println("---------------------------------------------------------------------------------------");
		

	}

}
