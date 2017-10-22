package ude.master.thesis.stance_detection.processor;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
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
import org.tartarus.snowball.ext.englishStemmer;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;

import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.EnhancedPlusPlusDependenciesAnnotation;
import edu.stanford.nlp.util.CoreMap;
import ude.master.thesis.stance_detection.util.BodySummerizer2;
import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;

/**
 * This class is the same as StanfordDependencyParser class but the body text
 * parts are changed
 * 
 * @author Razan
 *
 */
public class StanfordDependencyParserWithArguments {

	private static boolean pipeInitialized = false;
	private static StanfordCoreNLP pipeline;

	public static Annotation buildAnnotatedDoc(String text) {

		if (!pipeInitialized)
			initPipe();

		// create an empty Annotation just with the given text
		Annotation document = new Annotation(text);

		// run all Annotators on this text
		pipeline.annotate(document);

		return document;

	}

	private static void initPipe() {
		// creates a StanfordCoreNLP object, with POS tagging, lemmatization,
		// NER, parsing, and coreference resolution
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");
		pipeline = new StanfordCoreNLP(props);
		pipeInitialized = true;
	}

	public static List<SemanticGraph> buildDependencyGraph(Annotation document) {
		List<CoreMap> sentences = document.get(SentencesAnnotation.class);

		List<SemanticGraph> allSentencesDependencies = new ArrayList<SemanticGraph>();

		for (CoreMap sentence : sentences) {
			// this is the Stanford dependency graph of the current sentence
			SemanticGraph dependencies = sentence.get(EnhancedPlusPlusDependenciesAnnotation.class);
			allSentencesDependencies.add(dependencies);
		}
		return allSentencesDependencies;
	}

	public static List<String> lemmatize(Annotation document) {
		List<CoreMap> sentences = document.get(SentencesAnnotation.class);

		List<String> lemmas = new ArrayList<>();
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

	public static void main(String[] args) throws Exception {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.ARGUMENTED_BODIES_TRAIN,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.ARGUMENTED_BODIES_TEST);

		Map<Integer, String> trainingIdBoyMap = sddr.getTrainIdBodyMap();
		//generateRootDistFeaturesAndSave(trainingIdBoyMap, ProjectPaths.CSV_ROOT_DIST_FEATURE_ARG_TRAIN);

		HashMap<Integer, String> testIdBoyMap = sddr.getTestIdBodyMap();
		//generateRootDistFeaturesAndSave(testIdBoyMap, ProjectPaths.CSV_ROOT_DIST_FEATURE_ARG_TEST);
		saveRootDistFeaturesAsHashFile(ProjectPaths.CSV_ROOT_DIST_FEATURE_ARG_TRAIN,
				ProjectPaths.ROOT_DIST_FEATURE_ARG_TRAIN);
		saveRootDistFeaturesAsHashFile(ProjectPaths.CSV_ROOT_DIST_FEATURE_ARG_TEST, ProjectPaths.ROOT_DIST_FEATURE_ARG_TEST);

		/*
		 * String text3 =
		 * "Iraq Says Arrested Woman Is Not The Wife of ISIS Leader al-Baghdadi"
		 * ; Annotation doc = buildAnnotatedDoc(text3); List<SemanticGraph>
		 * graphs = buildDependencyGraph(doc); List<Integer> rootDistVec =
		 * getRooDistFeatureVec(graphs); System.out.println(rootDistVec);
		 */

		// Task: save dependencies already
		// saveDepsListsMaps();

		String txt = "Kim Jong-un has broken both of his ankles and is now in the hospital after undergoing "
				+ "surgery, a report in a South Korean newspaper claims. The North Korean leader has "
				+ "been missing for more than three weeks, fueling speculation about what could cause his "
				+ "unusual disappearance from the public eye. This rumor seems to confirm what North Korean "
				+ "state media had said on Thursday, when state broadcaster Korean Central Television "
				+ "reported that Kim was not feeling well, and was suffering from an uncomfortable "
				+ "physical condition. Have something to add to this story? Share it in the comments.";
		// System.out.println(getDependenciesAsTxtList(txt));
		// System.out.println(FeatureExtractor.getLemmatizedCleanStr("I am
		// happier than ever."));
	}

	private static void generateRootDistFeaturesAndSave(Map<Integer, String> trainingIdBoyMap, String filename)
			throws Exception {
		List<String[]> entries = new ArrayList<>();

		String[] header = new String[15];
		header[0] = "Body ID";
		for (int j = 1; j <= 5; j++) {
			header[j] = "disc_RootDist" + j;

		}

		for (int j = 1; j <= 5; j++) {
			header[j + 5] = "ref_RootDist" + j;
		}
		header[11] = "avg_disc_RootDist";
		header[12] = "avg_ref_RootDist";

		header[13] = "min_disc_RootDist";
		header[14] = "min_ref_RootDist";
		entries.add(header);

		int i = 0;
		for (Map.Entry<Integer, String> e : trainingIdBoyMap.entrySet()) {
			List<String> entry = new ArrayList<>();
			entry.add(Integer.toString(e.getKey()));

			// add the discuss root dist
			String bodyText = e.getValue();
			Annotation doc = buildAnnotatedDoc(bodyText);
			// System.out.println(e.getValue());
			List<SemanticGraph> graphs = buildDependencyGraph(doc);

			// for Discuss rootDist
			List<Double> discussRootDistVec = getDiscussRootDistFeatureVec(graphs);

			for (Double d : discussRootDistVec)
				entry.add(Double.toString(d));

			//System.out.println(entry.size());
			// add the refute root dist

			// for Refute rootDist
			List<Double> refuteRootDistVec = getRefuteRootDistFeatureVec(graphs);

			for (Double d : refuteRootDistVec)
				entry.add(Double.toString(d));
			
			//System.out.println(entry.size());

			double avgDisc = getAvgRootDist(discussRootDistVec);
			entry.add(Double.toString(avgDisc));

			double avgRef = getAvgRootDist(refuteRootDistVec);
			entry.add(Double.toString(avgRef));

			double minDisc = 1000.0;
			for (Double d : discussRootDistVec)
				if (d < minDisc)
					minDisc = d;

			entry.add(Double.toString(minDisc));

			double minRef = 1000.0;
			for (Double d : refuteRootDistVec)
				if (d < minRef)
					minRef = d;

			entry.add(Double.toString(minRef));

			if (entry.size() != 15)
				throw new Exception("not 14 features");

			entries.add(entry.toArray(new String[0]));
			i++;

			if (i % 1000 == 0)
				System.out.println("processed: " + i);
		}

		try (CSVWriter writer = new CSVWriter(new FileWriter(filename))) {
			writer.writeAll(entries);
		}

	}

	private static double getAvgRootDist(List<Double> discussRootDistVec) {
		double avgDisc = 0;
		int howMany = 0;
		for (Double d : discussRootDistVec)
			if (d < 1000) {
				avgDisc += d;
				howMany++;
			}
		if (howMany == 0)
			avgDisc = 1000.0;
		else
			avgDisc /= howMany;
		return avgDisc;
	}

	/**
	 * This represent 2 features: the distance from the root of a sentence to a
	 * refuting/discussing word saving the map as [
	 * <body_id>,["ref_RootDist","disc_RootDist"]] These 2 features are
	 * calculated for each sentence in the body (5 from the beginning and 3 at
	 * last) So the feature vector is of length 10
	 * 
	 * @param csvfilePath
	 * @param hashFileName
	 * @throws FileNotFoundException
	 * @throws ObjectExistsException
	 * @throws ClassNotFoundException
	 * @throws VersionMismatchException
	 * @throws IOException
	 */
	public static void saveRootDistFeaturesAsHashFile(String csvfilePath, String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, ArrayList<Double>> rootDistData = new FileHashMap<String, ArrayList<Double>>(hashFileName,
				FileHashMap.FORCE_OVERWRITE);
		CSVReader reader = null;
		reader = new CSVReader(new FileReader(csvfilePath));
		String[] line;
		line = reader.readNext();

		while ((line = reader.readNext()) != null) {

			ArrayList<Double> values = new ArrayList<>();
			for (int i = 1; i <= 14; i++)
				values.add(Double.valueOf(line[i]));

			rootDistData.put(line[0], values);
		}
		reader.close();

		// saving the map file
		rootDistData.save();
		rootDistData.close();

	}

	public static FileHashMap<String, ArrayList<Double>> loadRootDistFeaturesAsHashFile(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, ArrayList<Double>> rootDistData = new FileHashMap<String, ArrayList<Double>>(hashFileName,
				FileHashMap.RECLAIM_FILE_GAPS);
		return rootDistData;

	}

	private static List<Double> getDiscussRootDistFeatureVec(List<SemanticGraph> graphs) {

		List<String> discuss_words = Arrays.asList(FeatureExtractorWithModifiedBL.discussWordsJoined);

		List<Double> mins_dist = new ArrayList<>();

		for (SemanticGraph graph : graphs) {
			int minDiscussDist = 1000;

			IndexedWord root = graph.getFirstRoot(); // TODO: it it precise to
														// take only the first
														// root
			if (graph.getRoots().size() > 1)
				System.out.println("num of roots: " + graph.getRoots().size());

			// traverse the nodes in the graph
			for (int i = 1; i <= graph.size(); i++) {
				IndexedWord idxW = graph.getNodeByIndexSafe(i);
				if (idxW != null)
					if (idxW.word() != null)
						if ((discuss_words.contains(idxW.word().toLowerCase()))) {
							int dist = graph.getShortestDirectedPathEdges(root, graph.getNodeByIndex(i)).size();

							if (dist < minDiscussDist)
								minDiscussDist = dist;
						}
			}
			mins_dist.add((double) minDiscussDist);

		}

		// get the values in min_dist that are smaller than 1000 first
		List<Double> min_dist_red = new ArrayList<>();
		for (Double m : mins_dist) {
			if (m < 1000.0)
				min_dist_red.add(m);
		}
		if (min_dist_red.size() > 5)
			System.out.println("we got it >5!");		
		
		while (min_dist_red.size() > 5){
			min_dist_red.remove(min_dist_red.size()-1);
		}
		while (min_dist_red.size() < 5)
			min_dist_red.add(1000.0);

		return min_dist_red;
	}

	private static List<Double> getRefuteRootDistFeatureVec(List<SemanticGraph> graphs) {

		List<String> refuting_words = Arrays.asList(FeatureExtractorWithModifiedBL.refutingWords);

		List<Double> mins_dist = new ArrayList<>();
		// get the min rootDist for each graph (2 values vector for each)
		// System.out.println("graphs.size = " + graphs.size());
		for (SemanticGraph graph : graphs) {
			int minRefuteDist = 1000;

			IndexedWord root = graph.getFirstRoot();

			if (graph.getRoots().size() > 1)
				System.out.println("num of roots: " + graph.getRoots().size());

			// traverse the nodes in the graph
			for (int i = 1; i <= graph.size(); i++) {
				IndexedWord idxW = graph.getNodeByIndexSafe(i);

				if ((idxW != null))
					if (idxW.word() != null)
						if ((refuting_words.contains(idxW.word().toLowerCase()))) {
							int dist = graph.getShortestDirectedPathEdges(root, graph.getNodeByIndex(i)).size();

							if (dist < minRefuteDist)
								minRefuteDist = dist;
						}

			}
			mins_dist.add((double) minRefuteDist);

		}

		// get the values in min_dist that are smaller than 1000 first
		List<Double> min_dist_red = new ArrayList<>();
		for (Double m : mins_dist) {
			if (m < 1000.0)
				min_dist_red.add(m);
		}
		if (min_dist_red.size() > 5)
			System.out.println("we got it >5!");

		while (min_dist_red.size() > 5){
			min_dist_red.remove(min_dist_red.size()-1);
		}

		while (min_dist_red.size() < 5)
			min_dist_red.add(1000.0);

		return min_dist_red;
	}

	/**
	 * 
	 * @throws IOException
	 * @throws ObjectExistsException
	 * @throws ClassNotFoundException
	 * @throws VersionMismatchException
	 */
	public static void saveDepsListsMaps()
			throws IOException, ObjectExistsException, ClassNotFoundException, VersionMismatchException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.ARGUMENTED_BODIES_TRAIN,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.ARGUMENTED_BODIES_TEST);

		// ******************* Do training data ****************************
		// Use summarized bodies instead
		Map<Integer, String> trainingIdBoyMap = sddr.getTrainIdBodyMap();

		// Do the training bodies
		FileHashMap<Integer, List<String>> trainBDepsMap = new FileHashMap<Integer, List<String>>(
				ProjectPaths.TRAIN_ARG_BODIES_DEPS, FileHashMap.FORCE_OVERWRITE);

		for (Entry<Integer, String> b : trainingIdBoyMap.entrySet()) {

			List<String> deps = new ArrayList<>();
			deps = getDependenciesAsTxtList(b.getValue());

			trainBDepsMap.put(b.getKey(), deps);
		}
		trainBDepsMap.save();
		trainBDepsMap.close();

		// No need to Do the titles

		// ******************* Do Test data ****************************
		HashMap<Integer, String> testIdBoyMap = sddr.getTestIdBodyMap();

		// Do the training bodies
		FileHashMap<Integer, List<String>> testBDepsMap = new FileHashMap<Integer, List<String>>(
				ProjectPaths.TEST_ARG_BODIES_DEPS, FileHashMap.FORCE_OVERWRITE);
		for (Entry<Integer, String> b : testIdBoyMap.entrySet()) {
			List<String> deps = new ArrayList<>();
			deps = getDependenciesAsTxtList(b.getValue());

			testBDepsMap.put(b.getKey(), deps);
		}
		testBDepsMap.save();
		testBDepsMap.close();

		// We don't have to do the titles
	}

	/**
	 * 
	 * @param text
	 *            the part of the body with a distinct number of sentences
	 * @return a list of dependencies; a dependency for each sentence in the
	 *         text/part
	 */
	public static List<String> getDependenciesAsTxtList(String text) {
		Annotation doc = StanfordDependencyParserWithArguments.buildAnnotatedDoc(text);

		List<SemanticGraph> graphs = StanfordDependencyParserWithArguments.buildDependencyGraph(doc);

		List<String> depLists = new ArrayList<>();
		// Sentence / Graph Index
		for (SemanticGraph graph : graphs) {

			String depList = graph.toList();

			depLists.add(depList);
		}
		return depLists;
	}

}
