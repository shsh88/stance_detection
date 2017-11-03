package ude.master.thesis.stance_detection.featureengineering;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
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

import com.opencsv.CSVWriter;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import ude.master.thesis.stance_detection.util.BodySummerizer2;
import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.Tree;

public class SentencesLength {
	private static StanfordCoreNLP pipeline;

	public static void main(String[] args)
			throws IOException, ObjectExistsException, ClassNotFoundException, VersionMismatchException {

		// String test = "This Is a Truly Horrifying Story About a Spider";
		// System.out.println(findSentiment(test));

		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.PREPROCESSED_BODIES_TRAIN,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.PREPROCESSED_BODIES_TEST);

		Map<Integer, String> trainingSummIdBoyMap = sddr.getTrainIdBodyMap();
		HashMap<Integer, String> testSummIdBoyMap = sddr.getTestIdBodyMap();

		generateAndSaveBodiesSentenceLength(trainingSummIdBoyMap, testSummIdBoyMap, ProjectPaths.BODIES_SENTENCE_LENGTH,
				ProjectPaths.CSV_BODIES_SENTENCE_LENGTH);
	}

	/**
	 * Generate the sentiments for the body as for first sentences and then last
	 * 3 and at last sentences from the middle part
	 * 
	 * @param trainingSummIdBoyMap
	 * @param testSummIdBoyMap
	 * @param hashFilePath
	 * @param csvFilePath
	 * @throws IOException
	 * @throws ObjectExistsException
	 * @throws ClassNotFoundException
	 * @throws VersionMismatchException
	 */
	private static void generateAndSaveBodiesSentenceLength(Map<Integer, String> trainingSummIdBoyMap,
			HashMap<Integer, String> testSummIdBoyMap, String hashFilePath, String csvFilePath)
			throws IOException, ObjectExistsException, ClassNotFoundException, VersionMismatchException {
		if (pipeline == null)
			pipeline = initStanfordPipeline();

		FileHashMap<String, double[]> sentLen = new FileHashMap<String, double[]>(hashFilePath, FileHashMap.FORCE_OVERWRITE);

		List<String[]> entries = new ArrayList<>();

		String[] header = new String[3];
		header[0] = "body_id";

		header[1] = "sent_len_avg";
		header[2] = "sent_len_max";

		entries.add(header);

		Map<Integer, String> allBodyId = new HashMap<Integer, String>();
		allBodyId.putAll(trainingSummIdBoyMap);
		allBodyId.putAll(testSummIdBoyMap);

		for (Entry<Integer, String> b : allBodyId.entrySet()) {
			double[] features = new double[2];

			double avg_len = 0.0;
			int max_len = 0;

			String paragraph = b.getValue();

				Annotation annotation = pipeline.process(paragraph);
				int count = 0;
				for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
					int len = sentence.toString().length();
					avg_len += len;
					if (len > max_len) {
						max_len = len;
					}
					count++;
				}
			features[0] = avg_len / count;
			features[1] = max_len;
			sentLen.put(String.valueOf(b.getKey()), features);

			List<String> entry = new ArrayList<>();
			entry.add(String.valueOf(b.getKey()));
			for (double s : features) {
				entry.add(String.valueOf(s));
			}
			entries.add(entry.toArray(new String[0]));

		}

		try (CSVWriter writer = new CSVWriter(new FileWriter(csvFilePath))) {
			writer.writeAll(entries);
		}

		sentLen.save();
		sentLen.close();

	}

	private static StanfordCoreNLP initStanfordPipeline() {
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit, pos, lemma");
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		return pipeline;
	}

	public static FileHashMap<String, double[]> loadBodiesSentenceLengthAsHashFile(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, double[]> bSent = new FileHashMap<String, double[]>(hashFileName,
				FileHashMap.RECLAIM_FILE_GAPS);
		return bSent;
	}

}
