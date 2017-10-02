package ude.master.thesis.stance_detection.processor;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;
import java.util.Set;
import java.util.TreeSet;

public class FeatureExtractorWithModifiedBL {

	// TODO: These vocabulary needs stemming to be more precise, then needs to
	// be compared to words from input after stemming (e.g. no need to have
	// doubt & doubts together)
	public static String[] refutingWords = { "fake", "fraud", "hoax", "false", "deny", "denies", "not", "despite",
			"nope", "doubt", "doubts", "bogus", "debunk", "pranks", "retract" };

	// TODO: Needs investigation--> look at what words used in discussing
	// articles (Maybe find the words that shows more often / intersect)
	public static String[] discussWords = { "according", "maybe", "reporting", "reports", "say", "says", "claim",
			"claims", "purportedly", "investigating", "told", "tells", "allegedly", "validate", "verify" };

	public static String[] discussWordsJoined = { "according", "alleged", "allegedly", "apparently", "appear",
			"appears", "claim", "claims", "could", "evidently", "investigating", "largely", "likely", "mainly", "may",
			"maybe", "might", "mostly", "perhaps", "presumably", "probably", "purported", "purportedly", "reported",
			"reportedly", "rumor", "rumour", "rumors", "rumours", "rumored", "rumoured", "says", "say", "seem",
			"somewhat", "told", "tells",
			// "supposedly",
			"unconfirmed", "validate", "verify" };

	private static TreeSet<String> stopSet;

	private static Lemmatizer lemmatizer;
	private static final String STOPWORDS_FILE = "resources/stopwords.txt";

	public static String clean(String txt) {
		return txt.replaceAll("\\W+", " ").toLowerCase();

	}

	// This method calculate the feature for one record
	public static double getWordOverlapFeature(String headline, String body) {
		
		if(lemmatizer == null){
			lemmatizer = new Lemmatizer();
		}
		
		List<String> hLemmas = Arrays.asList(getLemmatizedCleanStr(headline).split("\\s+"));
		List<String> bLemmas = Arrays.asList(getLemmatizedCleanStr(body).split("\\s+"));
		
		

		List<String> headlineLem = removeStopWords(hLemmas);
		//there is no need to remove stop words from the body because there will be no match in the title
		//because we removed stop words from the title
		
		Set<String> intersectinSet = new HashSet<>(headlineLem);
		Set<String> bodySet = new HashSet<>(bLemmas);

		Set<String> UnionSet = new HashSet<>(headlineLem);

		intersectinSet.retainAll(bodySet);
		UnionSet.addAll(bodySet);

		return (double) intersectinSet.size() / (double) UnionSet.size();

	}

	// TODO Do we do this feature just for the headline ?
	public static int getRefutingFeature(String headline, String refutingWord) {

		String headlineLem = getLemmatizedCleanStr(headline);

		int f;
		if (headlineLem.contains(refutingWord))
			f = 1; // TODO may change this to add integers
		else
			f = 0;
		return f;
	}

	public static int calculatePolarity(String text) {

		int sum = 0;
		for (String polarityWord : refutingWords)
			if (text.contains(polarityWord))
				sum++;

		return sum % 2;
	}

	// TODO what about that it is done without lemmatization in baseline
	public static List<Integer> getBinaryCoOccurenceFeatures(String headline, String body) {
		int binCount = 0;
		int binCountEarly = 0;
		
		List<String> headlineLem = Arrays.asList(getLemmatizedCleanStr(headline).split(" "));
		for (String token : headlineLem) {
			if (body.contains(token)) 
											
				binCount++;

			if (body.length() >= 255) {
				if (body.substring(0, 255).contains(token))
					binCountEarly++;
				// TODO Do we really need to add this if the text length < 255
				// (the next else) ?
			} else {
				if (body.contains(token))
					binCountEarly++;
			}
		}

		List<Integer> f = new ArrayList<>();
		f.add(binCount);
		f.add(binCountEarly);
		return f;

	}

	// TODO I may remove binCountEarly 
	public static List<Integer> getBinaryCoOccurenceStopFeatures(String headline, String body) {
		
		if(lemmatizer == null)
			lemmatizer = new Lemmatizer();
		
		int binCount = 0;
		int binCountEarly = 0;

		List<String> cleanHead = removeStopWords(Arrays.asList(getLemmatizedCleanStr(headline).split("\\s+")));
		
		String cleanBody = getLemmatizedCleanStr(body);

		for (String token : cleanHead) {
			if (cleanBody.contains(token)) 
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
	
	public static int getSentenceBinaryCoOccurenceStopFeatures(String headline, String sentence) {
		if(lemmatizer == null)
			lemmatizer = new Lemmatizer();
		
		int binCount = 0;

		// String[] cleanHeadLine = cleanText(headline).split(" ");
		List<String> cleanHead = removeStopWords(Arrays.asList(getLemmatizedCleanStr(headline).split("\\s+")));

		String lemmBody = getLemmatizedCleanStr(sentence);

		for (String token : cleanHead) {
			if (lemmBody.contains(token)) 
				binCount++;			
		}
		return binCount;
	}

	private static List<String> getLowerCase(List<String> cleanHead) {
		List<String> lowerHead = new ArrayList<>();
		for(String c : cleanHead)
			lowerHead.add(c.toLowerCase());
		return lowerHead;
	}

	public static List<String> removeStopWords(List<String> tokens) {
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
	
	public static Set<String> removeStopwords(Set<String> words) {
		Set<String> wordsNoStop = new HashSet<>();

		for (String word : words) {
			if (word.isEmpty())
				continue;
			if (isStopword(word))
				continue; // remove stopwords
			wordsNoStop.add(word);
		}
		return wordsNoStop;
	}


	private static boolean isStopword(String word) {
		if (stopSet == null)
			try {
				initializeStopwords(STOPWORDS_FILE);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

		//if (word.length() < 2)
		//	return true;
		//if (word.charAt(0) >= '0' && word.charAt(0) <= '9')
			//return true; // remove numbers, "23rd", etc
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
	// TODO in baseline they did not do lemmatization. Add some way to do text
	// cleaning like the baseline
	//new: Added lemmatization to the headline	
	public static List<Integer> getCharGramsFeatures(String headline, String body, int size) {
		List<String> h = removeStopWords(Arrays.asList(getLemmatizedCleanStr(headline).split("\\s+")));

		// get the string back
		StringBuilder sb = new StringBuilder();
		for (String s : h) {
			sb.append(s);
			sb.append(" ");
		}
		headline = sb.toString().trim();
		List<String> grams = getCharGrams(headline, size);
		
		List<String> b = removeStopWords(Arrays.asList(getLemmatizedCleanStr(body).split("\\s+")));

		// get the string back
		StringBuilder sb2 = new StringBuilder();
		for (String s : b) {
			sb2.append(s);
			sb2.append(" ");
		}
		body = sb2.toString().trim();

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
			}
		}

		List<Integer> f = new ArrayList<>();
		f.add(gramHits);
		f.add(gramEarlyHits);
		f.add(gramFirstHits);
		// f.add(gramTailHits);

		return f;
	}
	public static int getSentenceCharGramsFeatures(String headline, String body, int size) {
		List<String> h = removeStopWords(Arrays.asList(getLemmatizedCleanStr(headline).split("\\s+")));

		// get the string back
		StringBuilder sb = new StringBuilder();
		for (String s : h) {
			sb.append(s);
			sb.append(" ");
		}
		headline = sb.toString().trim();
		List<String> grams = getCharGrams(headline, size);
		
		List<String> b = removeStopWords(Arrays.asList(getLemmatizedCleanStr(body).split("\\s+")));

		// get the string back
		StringBuilder sb2 = new StringBuilder();
		for (String s : b) {
			sb2.append(s);
			sb2.append(" ");
		}
		body = sb2.toString().trim();

		int gramHits = 0;
		
		for (String gram : grams) {
			if (body.contains(gram)) 
				gramHits++;
			
		}

		return gramHits;
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

	/**
	 * Returns a vector of 2 features
	 * 
	 * @param headline
	 * @param body
	 * @param size
	 * @return
	 */
	public static List<Integer> getNGramsFeatures(String headline, String body, int size) {

		List<String> h = removeStopWords(Arrays.asList(getLemmatizedCleanStr(headline).split("\\s+")));
		StringBuilder sb = new StringBuilder();
		for (String s : h) {
			sb.append(s);
			sb.append(" ");
		}
		headline = sb.toString().trim();
		List<String> grams = getNGrams(headline, size);
		
		List<String> b = removeStopWords(Arrays.asList(getLemmatizedCleanStr(body).split("\\s+")));

		// get the string back
		StringBuilder sb2 = new StringBuilder();
		for (String s : b) {
			sb2.append(s);
			sb2.append(" ");
		}
		body = sb2.toString().trim();
		
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
			} else {
				if (body.contains(gram)) {
					gramEarlyHits++;
				}
			}

		}

		List<Integer> f = new ArrayList<>();
		f.add(gramHits);
		f.add(gramEarlyHits);

		return f;

	}
	
	public static int getSentenceNGramsFeatures(String headline, String body, int size) {

		List<String> h = removeStopWords(Arrays.asList(getLemmatizedCleanStr(headline).split("\\s+")));
		StringBuilder sb = new StringBuilder();
		for (String s : h) {
			sb.append(s);
			sb.append(" ");
		}
		headline = sb.toString().trim();
		List<String> grams = getNGrams(headline, size);
		
		List<String> b = removeStopWords(Arrays.asList(getLemmatizedCleanStr(body).split("\\s+")));

		// get the string back
		StringBuilder sb2 = new StringBuilder();
		for (String s : b) {
			sb2.append(s);
			sb2.append(" ");
		}
		body = sb2.toString().trim();
		
		int gramHits = 0;

		for (String gram : grams) {
			if (body.contains(gram)) 
				gramHits++;
		}
		return gramHits;

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
		if(lemmatizer == null)
			lemmatizer = new Lemmatizer();
		
		List<String> strLem = lemmatizer.lemmatize(str);

		String lem = "";
		for (String w : strLem)
			lem += w + " ";
		return clean(lem.trim());
	}
	
	public static List<String> getLemmatizedCleanStrList(String str) {
		if(lemmatizer == null)
			lemmatizer = new Lemmatizer();
		
		List<String> strLem = lemmatizer.lemmatize(clean(str));

		return strLem;
	}

}
