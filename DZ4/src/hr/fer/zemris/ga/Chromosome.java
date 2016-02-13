package hr.fer.zemris.ga;

public interface Chromosome {

	public double getFitness();

	public void setFitness(double fitness);

	public int getSize();

	public double getValue(int index);
	
	public void setValue(int index, double value);
	
	public double[] getPoints();
	
	public int[] getBinary(int index);

}
