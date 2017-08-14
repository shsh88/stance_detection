package ude.master.thesis.stance_detection.processor;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;
import java.util.Set;
import java.util.TreeSet;

public class FeatureExtractor {

	// TODO: These vocabulary needs stemming to be more precise, then needs to
	// be compared to words from input after stemming (e.g. no need to have
	// doubt & doubts together)
	public static String[] refutingWords = { "fake", "fraud", "hoax", "false", "deny", "denies", "not", "despite",
			"nope", "doubt", "doubts", "bogus", "debunk", "pranks", "retract" };

	// TODO: Needs investigation--> look at what words used in discussing
	// articles (Maybe find the words that shows more often / intersect)
	public static String[] discussWords = { "according", "maybe", "reporting", "reports", "say", "says", "claim",
			"claims", "purportedly", "investigating", "told", "tells", "allegedly", "validate", "verify" };

	private static TreeSet<String> stopSet;
	private static final String STOPWORDS_FILE = "resources/stopwords.txt";

	public static String clean(String txt) {
		return txt.replaceAll("\\W+", " ").toLowerCase();

	}

	// This method calculate the feature for one record
	public static double getWordOverlapFeature(String headline, String body) {

		// TODO: What about numbers ? (e.g. 1.96$ --> 1, 96)
		List<String> headlineLem = new Lemmatizer().lemmatize(clean(headline));
		List<String> bodyLem = new Lemmatizer().lemmatize(clean(body));

		Set<String> intersectinSet = new HashSet<>(headlineLem);
		Set<String> bodySet = new HashSet<>(bodyLem);

		Set<String> UnionSet = new HashSet<>(headlineLem);

		intersectinSet.retainAll(bodySet);
		UnionSet.addAll(bodySet);

		return (double) intersectinSet.size() / (double) UnionSet.size();

	}

	// TODO Do we do this feature just for the headline ?
	public static int getRefutingFeature(String headline, String refutingWord) {

		List<String> headlineLem = new Lemmatizer().lemmatize(clean(headline));

		int f;
		if (headlineLem.contains(refutingWord))
			f = 1; // TODO may change this to add integers
		else
			f = 0;
		return f;
	}

	public static int calculatePolarity(String text) {

		List<String> textLem = new Lemmatizer().lemmatize(clean(text));

		int sum = 0;
		for (String polarityWord : refutingWords)
			if (textLem.contains(polarityWord))
				sum++;

		return sum % 2;
	}

	// TODO what about that it is done without lemmatization in baseline
	public static List<Integer> getBinaryCoOccurenceFeatures(String headline, String body) {
		int binCount = 0;
		int binCountEarly = 0;

		List<String> cleanHeadline = Arrays.asList(clean(headline).split(" "));

		String cleanBody = clean(body);

		for (String token : cleanHeadline) {
			if (cleanBody.contains(token)) // TODO traverse, won't we find this?
											// --> verse
				binCount++;

			if (cleanBody.length() >= 255) {
				if (cleanBody.substring(0, 255).contains(token))
					binCountEarly++;
				// TODO Do we really need to add this if the text length < 255 ?
			} else {
				if (cleanBody.contains(token))
					binCountEarly++;
			}
		}

		List<Integer> f = new ArrayList<>();
		f.add(binCount);
		f.add(binCountEarly);
		return f;

	}

	// TODO what about that it is done without lemmatization in baseline
	public static List<Integer> getBinaryCoOccurenceStopFeatures(String headline, String body) {
		int binCount = 0;
		int binCountEarly = 0;

		try {
			initializeStopwords(STOPWORDS_FILE);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		// String[] cleanHeadLine = cleanText(headline).split(" ");
		List<String> cleanHead = removeStopWords(Arrays.asList(clean(headline).split(" ")));

		String cleanBody = clean(body);

		for (String token : cleanHead) {
			if (cleanBody.contains(token)) // TODO traverse, won't we find this?
											// --> verse
				binCount++;
			if (cleanBody.length() >= 255) {
				if (cleanBody.substring(0, 255).contains(token))
					binCountEarly++;
			} else {
				if (cleanBody.contains(token))
					binCountEarly++;
			}
		}

		List<Integer> f = new ArrayList<>();
		f.add(binCount);
		f.add(binCountEarly);
		return f;
	}

	private static List<String> removeStopWords(List<String> tokens) {
		List<String> wordsNoStop = new ArrayList<>();

		for (String word : tokens) {
			if (word.isEmpty())
				continue;
			if (isStopword(word))
				continue; // remove stopwords
			wordsNoStop.add(word);
		}
		return wordsNoStop;
	}

	private static boolean isStopword(String word) {
		// if (word.length() < 2)
		// return true;
		// if (word.charAt(0) >= '0' && word.charAt(0) <= '9')
		// return true; // remove numbers, "23rd", etc
		if (stopSet.contains(word))
			return true;
		else
			return false;
	}

	private static void initializeStopwords(String stopFile) throws Exception {
		stopSet = new TreeSet<>();
		Scanner s = new Scanner(new FileReader(stopFile));
		while (s.hasNext())
			stopSet.add(s.next());
		s.close();
	}

	/**
	 * vector specifying the sum of how often character sequences of length
	 * 2,4,8,16 in the headline appear in the entire body, the first 100
	 * characters and the first 255 characters of the body.
	 * 
	 * @param headLine
	 * @param bodyLem
	 * @param size
	 * @return
	 */
	// TODO in baseline they didnot fo lemmatization. Add some way to do text
	// cleaning like the baseline
	public static List<Integer> getCharGramsFeatures(String headline, String body, int size) {
		List<String> h = removeStopWords(Arrays.asList(headline.split(" ")));

		// get the string back
		StringBuilder sb = new StringBuilder();
		for (String s : h) {
			sb.append(s);
			sb.append(" ");
		}
		headline = sb.toString().trim();
		List<String> grams = getCharGrams(headline, size);

		int gramHits = 0;
		int gramEarlyHits = 0;
		int gramFirstHits = 0;

		for (String gram : grams) {
			if (body.contains(gram)) {
				gramHits++;
			}
			if (body.length() >= 255) {
				if (body.substring(0, 255).contains(gram)) {
					gramEarlyHits++;
				}

				/*
				 * if (body.substring(body.length() - 255).contains(gram)) {
				 * gramTailHits++; }
				 */

			} else {
				if (body.contains(gram)) {
					gramEarlyHits++;
				}

				/*
				 * if (body.contains(gram)) { gramTailHits++; }
				 */
			}

			if (body.length() >= 100) {
				if (body.substring(0, 100).contains(gram)) {
					gramFirstHits++;
				}

				/*
				 * if (body.substring(body.length() - 100).contains(gram)) {
				 * gramTailHits++; }
				 */
			} else {
				if (body.contains(gram)) {
					gramFirstHits++;
				}
				/*
				 * if (body.contains(gram)) { gramTailHits++; }
				 */
			}
		}

		List<Integer> f = new ArrayList<>();
		f.add(gramHits);
		f.add(gramEarlyHits);
		f.add(gramFirstHits);
		// f.add(gramTailHits);

		return f;
	}

	/**
	 * get the char gram sequences from a given text by the given size
	 * 
	 * @param text
	 * @param size
	 * @return
	 */
	private static List<String> getCharGrams(String text, int size) {
		List<String> ret = new ArrayList<String>();
		for (int i = 0; i <= text.length() - size; i++) {
			String ngram = "";
			for (int j = i; j < i + size; j++)
				ngram += text.charAt(j);
			ngram.trim();
			ret.add(ngram);
		}
		return ret;
	}

	public static List<Integer> getNGramsFeatures(String headline, String body, int size) {

		List<String> grams = getNGrams(headline, size);

		int gramHits = 0;
		int gramEarlyHits = 0;

		for (String gram : grams) {
			if (body.contains(gram)) {
				gramHits++;
			}
			if (body.length() >= 255) {
				if (body.substring(0, 255).contains(gram)) {
					gramEarlyHits++;
				}

				/*
				 * if (bodyLem.substring(bodyLem.length() - 255).contains(gram))
				 * { gramTailHits++; }
				 */

			} else {
				if (body.contains(gram)) {
					gramEarlyHits++;
				}
				/*
				 * if (bodyLem.contains(gram)) { // TODO do we need to look in
				 * this // case gramTailHits++; }
				 */
			}

		}

		List<Integer> f = new ArrayList<>();
		f.add(gramHits);
		f.add(gramEarlyHits);

		return f;

	}

	private static List<String> getNGrams(String text, int n) {
		List<String> ret = new ArrayList<String>();
		String[] input = text.split(" ");
		for (int i = 0; i <= input.length - n; i++) {
			String ngram = "";
			for (int j = i; j < i + n; j++)
				ngram += input[j] + " ";
			ngram.trim();
			ret.add(ngram);
		}
		return ret;
	}

	public static String getLemmatizedCleanStr(String str) {
		List<String> strLem = new Lemmatizer().lemmatize(clean(str));

		String lem = "";
		for (String w : strLem)
			lem += w + " ";
		return lem.trim();
	}

}
