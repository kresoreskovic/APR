package hr.fer.zemris.ga;

import java.util.Random;

public class GeneticAlgorithm {

	public IFunction f;
	public int popSize;
	public double pm;
	public Random rand;
	public int generationLimit;
	public int elitism;
	public boolean binary;
	public int p;
	public int k;
	public int N;
	public int evalLimit;
	public boolean verbose;
	public static double eps = 10e-6;

	public GeneticAlgorithm(IFunction f, int popSize, double pm, Random rand, int generationLimit, int elitism,
			boolean binary, int p, int k, int N, int evalLimit, boolean verbose) {
		super();
		this.f = f;
		this.popSize = popSize;
		this.pm = pm;
		this.rand = rand;
		this.generationLimit = generationLimit;
		this.elitism = elitism;
		this.binary = binary;
		this.p = p;
		this.k = k;
		this.N = N;
		this.evalLimit = evalLimit;
		this.verbose = verbose;
	}

	public Chromosome generativeGA() {
		Chromosome[] population = GAUtil.initializePopulation(popSize, rand, binary, p, N);

		for (int i = 0; i < population.length; i++) {
			population[i].setFitness(f.fitness(population[i].getPoints()));
		}

		Chromosome prevBest = GAUtil.findBest(population);
		for (int generation = 1; generation <= generationLimit; generation++) {
			if (f.getEval() >= evalLimit) {
				if (verbose) {
					System.out.println("Evaluation limit reached!");
				}
				break;
			}
			Chromosome[] nextPopulation = new Chromosome[popSize];
			Chromosome best = GAUtil.findBest(population);
			if (best.getFitness() <= eps) {
				if (verbose) {
					System.out.println("Found minimum before iteration exhaustion!");
				}
				break;
			}
			// Elitizam.
			if (elitism == 1) {
				nextPopulation[0] = best;
			}
			// Generiraj preostale ili sve, ovisno o elitizmu.
			for (int i = elitism; i < popSize; i++) {
				// Selektiraj.
				Chromosome[] parents = GAUtil.proportionalSelect(population, rand, 2);
				// Krizaj.
				Chromosome child = null;
				child = GAUtil.arithmeticCrossover(parents[0], parents[1], rand);
				// Mutiraj.
				GAUtil.mutate(child, rand, pm);
				child.setFitness(f.fitness(child.getPoints()));
				nextPopulation[i] = child;
			}
			population = nextPopulation;

			if (prevBest.getFitness() > best.getFitness() && verbose) {
				System.out.println("Generation: " + generation + (generation < 1000 ? "\t" : "") + "\tFitness: "
						+ best.getFitness() + "\tChromosome: " + best.toString());
			}
			prevBest = best;
		}
		if (verbose) {
			System.out.println("Number of generative evaluations: " + f.getEval());
		}
		f.reset();
		return GAUtil.findBest(population);
	}

	public Chromosome eliminativeGA() {
		Chromosome[] population = GAUtil.initializePopulation(popSize, rand, binary, p, N);

		for (int i = 0; i < population.length; i++) {
			population[i].setFitness(f.fitness(population[i].getPoints()));
		}

		Chromosome prevBest = GAUtil.findBest(population);
		for (int generation = 1; generation <= generationLimit; generation++) {
			if (f.getEval() >= evalLimit) {
				if (verbose) {
					System.out.println("Evaluation limit reached!");
				}
				break;
			}
			Chromosome[] nextPopulation = new Chromosome[popSize];
			Chromosome best = GAUtil.findBest(population);
			if (best.getFitness() <= eps) {
				if (verbose) {
					System.out.println("Found minimum before iteration exhaustion!");
				}
				break;
			}
			// Troturnirska selekcija vraca indekse roditelja i najgore jedinke.
			int[] indices = GAUtil.tournamentSelect(population, rand, k);
			int worstIndex = indices[indices.length - 1];
			// Krizaj.
			Chromosome child = null;
			child = GAUtil.arithmeticCrossover(population[indices[0]], population[indices[1]], rand);
			// Mutiraj.
			GAUtil.mutate(child, rand, pm);
			child.setFitness(f.fitness(child.getPoints()));

			// Stavi sve osim najgore od tri selektirane.
			for (int i = 0; i < popSize; i++) {
				if (i == worstIndex) {
					nextPopulation[i] = child;
				} else {
					nextPopulation[i] = population[i];
				}
			}
			population = nextPopulation;

			if (prevBest.getFitness() > best.getFitness() && verbose) {
				System.out.println("Generation: " + generation + (generation < 1000 ? "\t" : "") + "\tFitness: "
						+ best.getFitness() + "\tChromosome: " + best.toString());
			}
			prevBest = best;
		}
		if (verbose) {
			System.out.println("Number of eliminative evaluations: " + f.getEval());
		}
		f.reset();
		return GAUtil.findBest(population);
	}

}
