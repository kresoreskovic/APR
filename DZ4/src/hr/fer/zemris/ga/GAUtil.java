package hr.fer.zemris.ga;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

public class GAUtil {

	private static class Pair implements Comparable<Pair> {
		double fitness;
		int index;

		public Pair(double fitness, int index) {
			this.fitness = fitness;
			this.index = index;
		}

		@Override
		public boolean equals(Object obj) {
			return super.equals(fitness) && super.equals(index);
		}

		@Override
		public int compareTo(Pair o) {
			if (this.fitness < o.fitness) {
				return -1;
			}
			if (this.fitness > o.fitness) {
				return 1;
			}
			return 0;
		}
	}

	public static Chromosome[] initializePopulation(int popSize, Random rand, boolean binary, int p, int N) {
		Chromosome[] population = new Chromosome[popSize];
		for (int i = 0; i < popSize; i++) {
			if (binary) {
				population[i] = new ChromosomeBinary(rand, N, p);
			} else {
				population[i] = new ChromosomeDouble(rand, N);
			}
		}
		return population;
	}

	public static Chromosome findBest(Chromosome[] population) {
		double minError = -1;
		Chromosome best = null;
		for (Chromosome chromosome : population) {
			if (chromosome.getFitness() < minError || minError == -1) {
				minError = chromosome.getFitness();
				best = chromosome;
			}
		}
		return best;
	}

	public static Chromosome[] proportionalSelect(Chromosome[] population, Random rand, int N) {
		Chromosome[] parents = new Chromosome[2];
		double sumFitness = 0;

		for (int i = 0; i < population.length; i++) {
			sumFitness += population[i].getFitness();
		}

		sumFitness = 1 / sumFitness;

		for (int parentIndex = 0; parentIndex < N; parentIndex++) {
			double limit = rand.nextDouble() * sumFitness;
			// Nadji pogodjenu jedinku.
			int chosen = 0;
			double upperLimit = population[chosen].getFitness();

			while (limit > upperLimit && chosen < population.length - 1) {
				chosen++;
				upperLimit += population[chosen].getFitness();
			}

			parents[parentIndex] = population[chosen];
		}
		return parents;
	}

	public static int[] tournamentSelect(Chromosome[] population, Random rand, int k) {
		Set<String> selected = new HashSet<>();
		while (selected.size() < k) {
			selected.add(Integer.toString(rand.nextInt(population.length)));
		}
		ArrayList<Pair> candidates = new ArrayList<>();
		for (String index : selected) {
			int i = Integer.parseInt(index);
			candidates.add(new Pair(population[i].getFitness(), i));
		}
		Collections.sort(candidates);
		int[] indices = new int[k];
		for (int i = 0; i < candidates.size(); i++) {
			indices[i] = candidates.get(i).index;
		}
		return indices;
	}

	public static Chromosome arithmeticCrossover(Chromosome parent1, Chromosome parent2, Random rand) {
		Chromosome child = new ChromosomeDouble(rand, parent1.getSize());
		for (int i = 0; i < parent1.getSize(); i++) {
			double a = parent1.getValue(i);
			double b = parent2.getValue(i);
			double r = rand.nextDouble();
			child.setValue(i, r * a + (1 - r) * b);
		}
		return child;
	}

	public static Chromosome uniformCrossover(Chromosome parent1, Chromosome parent2, Random rand) {
		int p = 5;
		Chromosome child = new ChromosomeBinary(rand, parent1.getSize(), p);
		for (int i = 0; i < parent1.getSize(); i++) {
			int[] p1 = parent1.getBinary(i);
			int[] p2 = parent2.getBinary(i);
			int[] sol = new int[p1.length];
			for (int j = 0; j < p1.length; j++) {
				if (p1[j] == p2[j]) {
					sol[j] = p1[j];
				} else {
					sol[j] = rand.nextDouble() < 0.5 ? p1[j] : p2[j];
				}
			}
			// Konvertiraj niz bitova u int, zatim u double.
			String tmp = "";
			for (int j : sol) {
				tmp += sol[j];
			}
			double value = Decoder.convertFromBinary(Integer.parseInt(tmp, 2), tmp.length());
			child.setValue(i, value);
		}
		return child;
	}

	public static void mutate(Chromosome child, Random rand, double pm) {
		for (int i = 0; i < child.getSize(); i++) {
			if (rand.nextDouble() <= pm) {
				child.setValue(i, rand.nextDouble() * (Decoder.hi - Decoder.lo) + Decoder.lo);
			}
		}
	}

	public static void binaryMutate(Chromosome child, Random rand, double pm) {
		for (int i = 0; i < child.getSize(); i++) {
			if (rand.nextDouble() <= pm) {
				int[] sol = child.getBinary(i);
				for (int j = 0; j < sol.length; j++) {
					if (rand.nextDouble() <= pm) {
						sol[j] = 1 - sol[j];
					}
				}
				// Konvertiraj niz bitova u int, zatim u double.
				String tmp = "";
				for (int j : sol) {
					tmp += sol[j];
				}
				double value = Decoder.convertFromBinary(Integer.parseInt(tmp, 2), tmp.length());
				child.setValue(i, value);
				// child.setValue(i, rand.nextDouble() * (Decoder.hi -
				// Decoder.lo) + Decoder.lo);
			}
		}
	}

}
