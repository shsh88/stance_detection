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

import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;

public class ArgumentsCounter {

	private static StanfordCoreNLP pipeline;
	public static void main(String[] args) throws IOException, ObjectExistsException, ClassNotFoundException, VersionMismatchException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.ARGUMENTED_BODIES_TRAIN,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.ARGUMENTED_BODIES_TEST);


		Map<Integer, String> trainIdBodyMap = sddr.getTrainIdBodyMap();
		HashMap<Integer, String> testIdBodyMap = sddr.getTestIdBodyMap();
		
		getArgCount(testIdBodyMap, trainIdBodyMap, 
				ProjectPaths.ARG_COUNT_TRAIN_TEST, ProjectPaths.CSV_ARG_COUNT_TRAIN_TEST);
	}

	private static void getArgCount(HashMap<Integer, String> testIdBodyMap,
			Map<Integer, String> trainIdBodyMap, String hashFileName, String csvFileName) throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException, IOException {
		FileHashMap<String, Integer> args = new FileHashMap<String, Integer>(hashFileName, FileHashMap.FORCE_OVERWRITE);

		List<String[]> entries = new ArrayList<>();

		String[] header = new String[15];
		header[0] = "Body ID";
		header[1] = "arg_num";

		entries.add(header);
		for( Entry<Integer, String> t : trainIdBodyMap.entrySet()){
			List<CoreMap> sentences = getSentencesFromParagraph(t.getValue());
			args.put(t.getKey().toString(), sentences.size());
			
			List<String> entry = new ArrayList<>();
			entry.add(t.getKey().toString());
			entry.add(String.valueOf(sentences.size()));
			entries.add(entry.toArray(new String[0]));
			
		}
		for( Entry<Integer, String> t : testIdBodyMap.entrySet()){
			List<CoreMap> sentences = getSentencesFromParagraph(t.getValue());
			args.put(t.getKey().toString(), sentences.size());
			
			List<String> entry = new ArrayList<>();
			entry.add(t.getKey().toString());
			entry.add(String.valueOf(sentences.size()));
			entries.add(entry.toArray(new String[0]));
			
		}
		
		try (CSVWriter writer = new CSVWriter(new FileWriter(csvFileName))) {
			writer.writeAll(entries);
		}

		args.save();
		args.close();
		
	}
	
	private static List<CoreMap> getSentencesFromParagraph(String paragraph) {
		if (pipeline == null)
			initStanfordPipeline();

		Annotation doc = new Annotation(paragraph);
		pipeline.annotate(doc);
		List<CoreMap> sentences = doc.get(SentencesAnnotation.class);
		return sentences;
	}
	private static void initStanfordPipeline() {
		Properties props;
		props = new Properties();
		props.put("annotators", "tokenize, ssplit, pos, lemma");

		pipeline = new StanfordCoreNLP(props);

	}
	
	public static FileHashMap<String, Integer> loadArgsCountFeaturesAsHashFiles(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, Integer> puncData = new FileHashMap<String, Integer>(hashFileName,
				FileHashMap.RECLAIM_FILE_GAPS);
		return puncData;
	}
}
