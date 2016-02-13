package hr.fer.zemris.ga;

public class Decoder {

	public static int lo = -50;
	public static int hi = 150;

	public Decoder(int lo, int hi) {
		Decoder.lo = lo;
		Decoder.hi = hi;
	}

	public static int convertToBinary(double x, int n) {
		return (int) ((x - lo) / (hi - lo) * (Math.pow(2, n - 1)));
	}

	public static double convertFromBinary(int b, int n) {
		return lo + b * (hi - lo) / (Math.pow(2, n - 1));
	}

}
