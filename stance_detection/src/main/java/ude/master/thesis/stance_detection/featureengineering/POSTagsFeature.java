package ude.master.thesis.stance_detection.featureengineering;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Map.Entry;

import org.clapper.util.misc.FileHashMap;
import org.clapper.util.misc.ObjectExistsException;
import org.clapper.util.misc.VersionMismatchException;

import com.opencsv.CSVWriter;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import ude.master.thesis.stance_detection.processor.FeatureExtractorWithModifiedBL;
import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;

public class POSTagsFeature {

	private static StanfordCoreNLP pipeline;

	public static void main(String[] args)
			throws IOException, ObjectExistsException, ClassNotFoundException, VersionMismatchException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.SUMMARIZED_TRAIN_BODIES,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.SUMMARIZED_TEST_BODIES);

		HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap = sddr
				.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TRAIN_BODIES));
		HashMap<Integer, Map<Integer, String>> testSummIdBoyMap = sddr
				.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TEST_BODIES));

		getBodiesPOSTags(trainingSummIdBoyMap, testSummIdBoyMap, ProjectPaths.POS_TAG_MID_PART_TRAIN_TEST,
				ProjectPaths.CSV_POS_TAG_MID_PART_TRAIN_TEST);

	}

	private static void getBodiesPOSTags(HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap,
			HashMap<Integer, Map<Integer, String>> testSummIdBoyMap, String hashFilePath, String csvFilePath)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {

		FileHashMap<String, String> posTags = new FileHashMap<String, String>(hashFilePath,
				FileHashMap.FORCE_OVERWRITE);

		List<String[]> entries = new ArrayList<>();

		String[] header = new String[2];
		header[0] = "body_id";

		header[1] = "pos_tokens";

		entries.add(header);

		Map<Integer, Map<Integer, String>> allBodyId = new HashMap<Integer, Map<Integer, String>>();
		allBodyId.putAll(trainingSummIdBoyMap);
		allBodyId.putAll(testSummIdBoyMap);

		for (Entry<Integer, Map<Integer, String>> b : allBodyId.entrySet()) {
			String posStr = "";
			for (Entry<Integer, String> part : b.getValue().entrySet()) {
				if (part.getKey() == 2) {
					if (!part.getValue().isEmpty())
						posStr += getPosTagsFromText(part.getValue());
				}
			}

			posTags.put(String.valueOf(b.getKey()), posStr);

			List<String> entry = new ArrayList<>();
			entry.add(String.valueOf(b.getKey()));
			entry.add(String.valueOf(posStr));
			entries.add(entry.toArray(new String[0]));

		}

		try (CSVWriter writer = new CSVWriter(new FileWriter(csvFilePath))) {
			writer.writeAll(entries);
		}

		posTags.save();
		posTags.close();

	}

	private static String getPosTagsFromText(String text) {

		if (pipeline == null)
			pipeline = initStanfordPipeline();

		Annotation document = new Annotation(text);
		pipeline.annotate(document);
		List<CoreMap> sentences = document.get(SentencesAnnotation.class);

		String tokenPOS = "";
		for (CoreMap sentence : sentences) {
			for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
				String lemma = token.lemma();
				String pos;
				if (!FeatureExtractorWithModifiedBL.isStopword(lemma)) {
					pos = token.get(PartOfSpeechAnnotation.class);
					tokenPOS += pos + "_" + lemma + " ";
				}

			}
		}

		return tokenPOS;
	}

	private static StanfordCoreNLP initStanfordPipeline() {
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit, pos, lemma");
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		return pipeline;
	}
	
	public static FileHashMap<String, String> loadPOSTagsAsHashFiles(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, String> posTags = new FileHashMap<String, String>(hashFileName,
				FileHashMap.RECLAIM_FILE_GAPS);
		return posTags;
	}

}
