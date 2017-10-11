package ude.master.thesis.stance_detection.processor;

import static org.simmetrics.builders.StringDistanceBuilder.with;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import org.clapper.util.misc.FileHashMap;
import org.clapper.util.misc.ObjectExistsException;
import org.clapper.util.misc.VersionMismatchException;
import org.simmetrics.StringDistance;
import org.simmetrics.StringMetric;
import org.simmetrics.metrics.CosineSimilarity;
import org.simmetrics.metrics.StringMetrics;
import org.simmetrics.simplifiers.Simplifiers;
import org.simmetrics.tokenizers.Tokenizers;

import com.opencsv.CSVWriter;

import edu.mit.jwi.IDictionary;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.POSTaggerAnnotator;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

import ude.master.thesis.stance_detection.util.LeskGlossOverlaps;
import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;
import ude.master.thesis.stance_detection.util.TitleAndBodyTextPreprocess;

public class SimilarityFeaturesWithArguments {

	private static StringDistance cosSimMetric;
	private static StringMetric cosSimStringMetric;
	private LeskGlossOverlaps lgo;

	StanfordCoreNLP pipeline;

	private static List<List<String>> trainingStances = new ArrayList<>();
	private static List<List<String>> testStances = new ArrayList<List<String>>();
	private static Map<Integer, String> trainingIdBoyMap = new HashMap<>();
	private static HashMap<Integer, String> testIdBodyMap = new HashMap<>();

	private void initCosSimilarityMetric() {
		System.out.println("init metric");
		// TreeSet<String> stopSet =
		// initializeStopwords("resources/stopwords.txt");

		// Set<String> commonWords = Sets.newHashSet(stopSet);
		cosSimMetric = with(new CosineSimilarity<String>()).simplify(Simplifiers.toLowerCase())
				.simplify(Simplifiers.removeNonWord()).tokenize(Tokenizers.whitespace()).tokenize(Tokenizers.qGram(5))
				.build();
	}

	private void initCosSimilarityStringMetric() {
		System.out.println("init metric");
		cosSimStringMetric = StringMetrics.cosineSimilarity();
	}

	private void initLesk() throws IOException {
		IDictionary dict = LeskGlossOverlaps.getDictionary();
		lgo = new LeskGlossOverlaps(dict);
		lgo.useStopList(true);
		lgo.useLemmatiser(true);
	}

	public void getCosSimFeatureVector(List<List<String>> stances, Map<Integer, String> trainingIdBoyMap,
			String filepath) throws Exception {
		// HashMap to save values
		FileHashMap<String, Double> cosSims = new FileHashMap<String, Double>(filepath, FileHashMap.FORCE_OVERWRITE);

		if (cosSimStringMetric == null)
			initCosSimilarityStringMetric();

		List<String[]> entries = new ArrayList<>();
		List<String> csvHeader = new ArrayList<>();
		csvHeader.add("title");
		csvHeader.add("Body ID");
		csvHeader.add("Stance");

		csvHeader.add("cos_sim");

		entries.add(csvHeader.toArray(new String[0]));

		for (List<String> s : stances) {
			List<String> entry = new ArrayList<>();
			String title = s.get(0);
			entry.add(title);
			entry.add(s.get(1));
			entry.add(s.get(2));

			String body = trainingIdBoyMap.get(Integer.valueOf(s.get(1)));

			Double fAll = getCosSimfeatureAll(title, body);

			entry.add(fAll.toString());

			if (entry.size() != 4)
				throw new Exception("not 1 features");

			cosSims.put(title + s.get(1), fAll);

			entries.add(entry.toArray(new String[0]));
		}

		cosSims.save();
		cosSims.close();

		CSVWriter writer = new CSVWriter(new FileWriter(filepath + ".csv"));
		writer.writeAll(entries);
		writer.flush();
		writer.close();
		System.out.println("saved saved saved");

	}

	private Double getCosSimfeatureAll(String title, String body) {
		return calcDistance(title, body);
	}

	private double calcDistance(String title, String str2) {
		title = FeatureExtractorWithModifiedBL.getLemmatizedCleanStr(title);
		List<String> h = FeatureExtractorWithModifiedBL.removeStopWords(Arrays.asList(title.split("\\s+")));
		// get the string back
		StringBuilder sb = new StringBuilder();
		for (String s : h) {
			sb.append(s);
			sb.append(" ");
		}
		title = sb.toString().trim();

		str2 = FeatureExtractorWithModifiedBL.getLemmatizedCleanStr(str2);
		List<String> str2L = FeatureExtractorWithModifiedBL.removeStopWords(Arrays.asList(str2.split("\\s+")));
		// get the string back
		StringBuilder sb1 = new StringBuilder();
		for (String s : str2L) {
			sb1.append(s);
			sb1.append(" ");
		}
		str2 = sb1.toString().trim();

		// return cosSimMetric.distance(title, str2);
		return cosSimStringMetric.compare(title, str2);
	}

	public void getLeskOverlapFeatureVector(List<List<String>> stances, Map<Integer, String> trainingIdBoyMap,
			String filepath) throws Exception {
		if (lgo == null)
			initLesk();

		// HashMap to save values
		FileHashMap<String, Double> leskOverlap = new FileHashMap<String, Double>(filepath,
				FileHashMap.FORCE_OVERWRITE);

		List<String[]> entries = new ArrayList<>();
		List<String> csvHeader = new ArrayList<>();
		csvHeader.add("title");
		csvHeader.add("Body ID");
		csvHeader.add("Stance");

		csvHeader.add("lesk");

		entries.add(csvHeader.toArray(new String[0]));

		for (List<String> s : stances) {
			List<String> entry = new ArrayList<>();
			String title = s.get(0);
			entry.add(title);
			entry.add(s.get(1));
			entry.add(s.get(2));

			String body = trainingIdBoyMap.get(Integer.valueOf(s.get(1)));

			Double fAll = getLeskfeatureAll(title, body);

			entry.add(fAll.toString());

			if (entry.size() != 4)
				throw new Exception("not 1 features");

			leskOverlap.put(title + s.get(1), fAll);

			entries.add(entry.toArray(new String[0]));
		}

		leskOverlap.save();
		leskOverlap.close();

		CSVWriter writer = new CSVWriter(new FileWriter(filepath + ".csv"));
		writer.writeAll(entries);
		writer.flush();
		writer.close();
		System.out.println("saved saved saved");

	}

	private Double getLeskfeatureAll(String title, String body) {
		return lgo.overlap(title, body);
	}

	public static FileHashMap<String, Double> loadCosSimFeaturesAsHashFile(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, Double> cosSims = new FileHashMap<String, Double>(hashFileName,
				FileHashMap.FORCE_OVERWRITE);
		return cosSims;
	}

	public static FileHashMap<String, Double> loadLeskFeaturesAsHashFile(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, Double> leskOverlap = new FileHashMap<String, Double>(hashFileName,
				FileHashMap.FORCE_OVERWRITE);
		return leskOverlap;
	}

	public static void loadData() throws IOException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.ARGUMENTED_BODIES_TRAIN,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.ARGUMENTED_BODIES_TEST);

		trainingIdBoyMap = sddr.getTrainIdBodyMap();
		testIdBodyMap = sddr.getTestIdBodyMap();

		trainingStances = sddr.getTrainStances();

		testStances = sddr.getTestStances();
	}

	public static void main(String[] args) throws Exception {
		loadData();
		SimilarityFeaturesWithArguments sf = new SimilarityFeaturesWithArguments();
		sf.getCosSimFeatureVector(trainingStances, trainingIdBoyMap, ProjectPaths.TRAIN_ARG_COS_SIM_STRMET_PATH);
		sf.getCosSimFeatureVector(testStances, testIdBodyMap, ProjectPaths.TEST_ARG_COS_SIM_STRMET_PATH);

		sf.getLeskOverlapFeatureVector(trainingStances, trainingIdBoyMap, ProjectPaths.TRAIN_ARG_LESK_PATH);
		sf.getLeskOverlapFeatureVector(testStances, testIdBodyMap, ProjectPaths.TEST_ARG_LESK_PATH);
	}

}
