package ude.master.thesis.stance_detection.processor;

import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import ude.master.thesis.stance_detection.util.FNCConstants;
import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.NotSerializableException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.Set;

import org.clapper.util.misc.FileHashMap;
import org.clapper.util.misc.ObjectExistsException;
import org.clapper.util.misc.VersionMismatchException;

import com.opencsv.CSVReader;

/**
 * 
 * @author Razan
 *
 */
public class Lemmatizer {

	private StanfordCoreNLP pipeline;
	private Porter porter;

	public Lemmatizer() {
		// Create StanfordCoreNLP object properties, with POS tagging
		// (required for lemmatization), and lemmatization
		Properties props;
		props = new Properties();
		props.put("annotators", "tokenize, ssplit, pos, lemma");

		this.pipeline = new StanfordCoreNLP(props);
	}

	/**
	 * Method to perform lemmatization on a given text
	 * 
	 * @param documentText
	 *            the text to lemmatize
	 * @return a List with all the words' lemmas
	 */
	public List<String> lemmatize(String documentText) {
		List<String> lemmas = new ArrayList<>();
		// Create an empty Annotation just with the given text
		Annotation document = new Annotation(documentText);
		// run all Annotators on this text
		this.pipeline.annotate(document);
		// Iterate over all of the sentences found
		List<CoreMap> sentences = document.get(SentencesAnnotation.class);
		for (CoreMap sentence : sentences) {
			// Iterate over all tokens in a sentence
			for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
				// Retrieve and add the lemma for each word into the
				// list of lemmas
				lemmas.add(token.get(LemmaAnnotation.class));
			}
		}
		return lemmas;
	}

	/**
	 * Get the lemmas with indexes after removing stop words
	 * 
	 * @param documentText
	 * @return
	 */
	public Map<String, Integer> lemmatizeWithIdx(String documentText) {
		if (porter == null)
			porter = new Porter();
		// String cleanTxt = FeatureExtractor.clean(documentText);

		Map<String, Integer> lemmas = new HashMap<>();
		// Create an empty Annotation just with the given text
		Annotation document = new Annotation(documentText);
		// run all Annotators on this text
		this.pipeline.annotate(document);
		// Iterate over all of the sentences found
		List<CoreMap> sentences = document.get(SentencesAnnotation.class);
		int i = 0; // for sentences
		int j = 0; // for tokens
		// System.out.println("ssize " + sentences.size());
		for (CoreMap sentence : sentences) {
			// Iterate over all tokens in a sentence
			for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
				// Retrieve and add the lemma for each word into the
				// list of lemmas
				String lemma = token.get(LemmaAnnotation.class);
				lemma = FeatureExtractor.clean(lemma).trim();

				try {
					if (FeatureExtractor.isStopword(lemma) || lemma.isEmpty())
						continue;
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}

				lemmas.put("" + porter.stripAffixes(lemma) + "," + i + "," + token.index(), j);
				j++;
			}
			i++;
		}
		return lemmas;
	}

	public static void lemmatizeDataAndSavInHashFile()
			throws IOException, ObjectExistsException, ClassNotFoundException, VersionMismatchException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.SUMMARIZED_TRAIN_BODIES,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.SUMMARIZED_TEST_BODIES);

		HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap = sddr
				.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TRAIN_BODIES));
		HashMap<Integer, Map<Integer, String>> testSummIdBoyMap = sddr
				.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TEST_BODIES));

		List<List<String>> trainingStances = sddr.getTrainStances();

		List<List<String>> testStances = sddr.getTestStances();

		// lemmatize and save titles (test & train)
		Set<String> titles = new HashSet<>();
		for (List<String> s : trainingStances) {
			titles.add(s.get(0));
		}
		for (List<String> s : testStances) {
			titles.add(s.get(0));
		}
		saveTitiesLemmasAsHashFiles(titles, ProjectPaths.TITLES_LEMMAS);
		saveBodiesLemmasAsHashFile(trainingSummIdBoyMap, testSummIdBoyMap, ProjectPaths.BODIES_LEMMAS);
	}

	/**
	 * This method isused to lemmatize body as complete, not parts
	 * Titles don't need to be lemmatized here.. it's the same no change to ProjectPaths.TITLES_LEMMAS
	 * @throws IOException
	 * @throws ObjectExistsException
	 * @throws ClassNotFoundException
	 * @throws VersionMismatchException
	 */
	public static void lemmatizeDataASWholeAndSavInHashFile()
			throws IOException, ObjectExistsException, ClassNotFoundException, VersionMismatchException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.ARGUMENTED_BODIES_TRAIN,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.ARGUMENTED_BODIES_TEST);

		Map<Integer, String> trainingIdBoyMap = sddr.getTrainIdBodyMap();
		HashMap<Integer, String> testIdBoyMap = sddr.getTestIdBodyMap();

		saveBodiesAsWholeLemmasAsHashFile(trainingIdBoyMap, testIdBoyMap, ProjectPaths.ARGS_BODIES_LEMMAS);
	}

	private static void saveBodiesAsWholeLemmasAsHashFile(Map<Integer, String> trainingIdBoyMap,
			HashMap<Integer, String> testIdBoyMap, String argsBodiesLemmasPath) throws NotSerializableException, IOException, ObjectExistsException, ClassNotFoundException, VersionMismatchException {
		FileHashMap<Integer, String> bodiesLemmas = new FileHashMap<>(argsBodiesLemmasPath,
				FileHashMap.FORCE_OVERWRITE);
		Lemmatizer lemm = new Lemmatizer();
		
		for(Entry<Integer, String> e: trainingIdBoyMap.entrySet()){
		    String bodyLemmas = concatToText(lemm.lemmatize(e.getValue()));
			bodiesLemmas.put(e.getKey(), bodyLemmas);
		}
		
		for(Entry<Integer, String> e: testIdBoyMap.entrySet()){
		    String bodyLemmas = concatToText(lemm.lemmatize(e.getValue()));
			bodiesLemmas.put(e.getKey(), bodyLemmas);
		}
		bodiesLemmas.save();
		bodiesLemmas.close();
		
	}

	private static void saveBodiesLemmasAsHashFile(HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap,
			HashMap<Integer, Map<Integer, String>> testSummIdBoyMap, String bodiesLemmasPath)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {

		FileHashMap<Integer, Map<Integer, String>> bodiesLemmas = new FileHashMap<>(bodiesLemmasPath,
				FileHashMap.FORCE_OVERWRITE);
		Lemmatizer lemm = new Lemmatizer();

		for (Entry<Integer, Map<Integer, String>> e : trainingSummIdBoyMap.entrySet()) {
			Map<Integer, String> partsMap = new HashMap<>();
			for (int i = 1; i <= 3; i++) {
				String partLemma = concatToText(lemm.lemmatize(e.getValue().get(i)));
				partsMap.put(i, partLemma);
			}
			bodiesLemmas.put(e.getKey(), partsMap);
		}

		for (Entry<Integer, Map<Integer, String>> e : testSummIdBoyMap.entrySet()) {
			Map<Integer, String> partsMap = new HashMap<>();
			for (int i = 1; i <= 3; i++) {
				String partLemma = concatToText(lemm.lemmatize(e.getValue().get(i)));
				partsMap.put(i, partLemma);
			}
			bodiesLemmas.put(e.getKey(), partsMap);
		}
		bodiesLemmas.save();
		bodiesLemmas.close();

	}

	private static String concatToText(List<String> titleLemma) {
		String lem = "";
		for (String w : titleLemma)
			lem += w + " ";
		return lem.trim();
	}

	public static void saveTitiesLemmasAsHashFiles(Set<String> titles, String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, String> titlesLemmas = new FileHashMap<String, String>(hashFileName,
				FileHashMap.FORCE_OVERWRITE);

		Lemmatizer lemm = new Lemmatizer();

		for (String t : titles) {
			String titleLemma = concatToText(lemm.lemmatize(t));
			titlesLemmas.put(t, titleLemma);
		}

		// saving the map file
		titlesLemmas.save();
		titlesLemmas.close();

	}

	public static FileHashMap<String, String> loadTitlesLemmasAsHashFiles(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, String> titlesLemmas = new FileHashMap<String, String>(hashFileName,
				FileHashMap.RECLAIM_FILE_GAPS);
		return titlesLemmas;
	}

	public static FileHashMap<Integer, Map<Integer, String>> loadBodiesLemmasAsHashFiles(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<Integer, Map<Integer, String>> bodiesLemmas = new FileHashMap<Integer, Map<Integer, String>>(
				hashFileName, FileHashMap.RECLAIM_FILE_GAPS);
		return bodiesLemmas;
	}
	public static FileHashMap<Integer, String> loadBodiesAsWholeLemmasAsHashFiles(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<Integer, String> bodiesLemmas = new FileHashMap<Integer, String>(
				hashFileName, FileHashMap.RECLAIM_FILE_GAPS);
		return bodiesLemmas;
	}

	public static void main(String[] args)
			throws ObjectExistsException, ClassNotFoundException, VersionMismatchException, IOException {
		//Lemmatizer lemm = new Lemmatizer();

		String str = "(NEWSER) – Wonder how long a Quarter Pounder with cheese can "
				+ "last? Two Australians say they bought a few McDonald's burgers for "
				+ "friends back in 1995, when they were teens, and one of the friends "
				+ "never showed up. So the kid's burger went uneaten—and stayed that way, "
				+ "Australia's News Network reports. \"We’re pretty sure it’s the oldest"
				+ " burger in the world,\" says one of the men, Casey Dean. Holding onto "
				+ "the burger for their friend \"started off as a joke,\" he adds, but \""
				+ "the months became years and now, 20 years later, it looks the same"
				+ " as it did the day we bought it, perfectly preserved in its original "
				+ "wrapping.\" Dean and his burger-buying mate, Eduard Nitz, even took "
				+ "the burger on Australian TV show The Project last night and \"showed"
				+ " off the mold-free specimen,\" News 9 reports. The pair offered to "
				+ "take a bite of it for charity but were dissuaded by the show's hosts. "
				+ "They've also started a Facebook page for the burger called \"Can This "
				+ "20 Year Old Burger Get More Likes Than Kanye West?\" that has more than "
				+ "4,044 likes as of this writing. And they're selling an iTunes song, "
				+ "\"Free the Burger,\" for $1.69, and giving proceeds to the charity "
				+ "Beyond Blue, which helps Australians battle anxiety and depression. "
				+ "(A few years ago, a man sold a 20-year-old bottle of McDonald's "
				+ "McJordan sauce for $10,000. Here's why Mickey D's food seemingly " + "never decays.)";
		// String resultString = str.replaceAll("\\W", " ").toLowerCase();
		// String resultString = cleanText(str);
		// System.out.println(lemm.lemmatize("Next-generation Apple iPhones
		// features leaked: it's not good"));
		//System.out.println(lemm.lemmatize(str));
		//lemmatizeDataAndSavInHashFile();
		lemmatizeDataASWholeAndSavInHashFile();
	}
}
