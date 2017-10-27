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

public class SentimentAnalysis {
	private static StanfordCoreNLP pipeline;

	public static void main(String[] args)
			throws IOException, ObjectExistsException, ClassNotFoundException, VersionMismatchException {

		// String test = "This Is a Truly Horrifying Story About a Spider";
		// System.out.println(findSentiment(test));

		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.SUMMARIZED_TRAIN_BODIES2_WITH_MID,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.SUMMARIZED_TEST_BODIES2_WITH_MID);

		HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap = sddr
				.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TRAIN_BODIES2_WITH_MID));
		HashMap<Integer, Map<Integer, String>> testSummIdBoyMap = sddr
				.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TEST_BODIES2_WITH_MID));

		List<List<String>> trainStances = sddr.getTrainStances();
		List<List<String>> testStances = sddr.getTestStances();

		generateAndsaveTitleSentiments(trainStances, testStances, ProjectPaths.TITLES_SENIMENTS,
				ProjectPaths.CSV_TITLES_SENIMENTS);

		generateAndSaveBodiesSentiments(trainingSummIdBoyMap, testSummIdBoyMap, ProjectPaths.BODIES_SENIMENTS,
				ProjectPaths.CSV_BODIES_SENIMENTS);
	}

	/**
	 * Generate the sentiments for the body as for first  sentences and then last 3 and at last sentences
	 * from the middle part
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
	private static void generateAndSaveBodiesSentiments(HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap,
			HashMap<Integer, Map<Integer, String>> testSummIdBoyMap, String hashFilePath, String csvFilePath)
			throws IOException, ObjectExistsException, ClassNotFoundException, VersionMismatchException {
		FileHashMap<String, ArrayList<Integer>> senti = new FileHashMap<String, ArrayList<Integer>>(hashFilePath,
				FileHashMap.FORCE_OVERWRITE);

		List<String[]> entries = new ArrayList<>();

		String[] header = new String[10];
		header[0] = "body_id";
		for (int i = 1; i < 9; i++)
			header[i] = "senti_" + i;
		
		header[9] = "senti_mid";

		entries.add(header);  
		
		Map<Integer, Map<Integer, String>> allBodyId = new HashMap<Integer, Map<Integer, String>>();
		allBodyId.putAll(trainingSummIdBoyMap);
		allBodyId.putAll(testSummIdBoyMap);
		
		for(Entry<Integer, Map<Integer, String>> b:allBodyId.entrySet()){
			ArrayList<Integer> sentiments = new ArrayList<>();
			
			for(Entry<Integer, String> parts: b.getValue().entrySet()){
				if(parts.getKey() != 2){
					sentiments.addAll(findSentiment(parts.getValue()));
					
					if(parts.getKey() == 1)
						while(sentiments.size() < BodySummerizer2.NUM_SENT_BEG)
							sentiments.add(-1); //no sentiment
					
					if(parts.getKey() == 3)
						while(sentiments.size() < BodySummerizer2.NUM_SENT_BEG + BodySummerizer2.NUM_SENT_END)
							sentiments.add(-1); //no sentiment
				}
			}
			
			if(sentiments.size() != 8)
				System.out.println("no 8 !");
			
			String midPart = b.getValue().get(2);
			sentiments.addAll(findSentiment(midPart));
			
			senti.put(String.valueOf(b.getKey()), sentiments);
			
			List<String> entry = new ArrayList<>();
			for(Integer s:sentiments){
				entry.add(String.valueOf(s));
			}
			entries.add(entry.toArray(new String[0]));
			
		}
		

		try (CSVWriter writer = new CSVWriter(new FileWriter(csvFilePath))) {
			writer.writeAll(entries);
		}

		senti.save();
		senti.close();

	}

	private static void generateAndsaveTitleSentiments(List<List<String>> trainStances, List<List<String>> testStances,
			String hashFilePath, String csvFilePath) throws FileNotFoundException, ObjectExistsException,
			ClassNotFoundException, VersionMismatchException, IOException {
		FileHashMap<String, Integer> senti = new FileHashMap<String, Integer>(hashFilePath,
				FileHashMap.FORCE_OVERWRITE);

		List<String[]> entries = new ArrayList<>();

		String[] header = new String[2];
		header[0] = "title";
		header[1] = "t_senti";

		entries.add(header);
		
		Set<String> allTitles = new HashSet<String>(); 
		for (List<String> s  : trainStances){
			allTitles.add(s.get(0));
		}
		
		for (List<String> s  : testStances){
			allTitles.add(s.get(0));
		}
		
		for(String t:allTitles){
			int titleSenti = findSentiment(t).get(0);
			senti.put(t, titleSenti);
			
			List<String> entry = new ArrayList<>();
			entry.add(t);
			entry.add(String.valueOf(titleSenti));
			entries.add(entry.toArray(new String[0]));
		}

		try (CSVWriter writer = new CSVWriter(new FileWriter(csvFilePath))) {
			writer.writeAll(entries);
		}

		senti.save();
		senti.close();

	}

	public static ArrayList<Integer> findSentiment(String paragraph) {
		ArrayList<Integer> sentiments = new ArrayList<>();

		if(pipeline == null)
			pipeline = initSentimentPipeline();

		if (paragraph != null && paragraph.length() > 0) {
			Annotation annotation = pipeline.process(paragraph);
			for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
				Tree tree = sentence.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);
				int sentiment = RNNCoreAnnotations.getPredictedClass(tree);
				sentiments.add(sentiment);
			}
		}
		return sentiments;
	}

	private static StanfordCoreNLP initSentimentPipeline() {
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit, parse, sentiment");
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		return pipeline;
	}
	
	public static FileHashMap<String, Integer> loadTitleSentimentsAsHashFile(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, Integer> tSent = new FileHashMap<String, Integer>(hashFileName,
				FileHashMap.RECLAIM_FILE_GAPS);
		return tSent;
	}
	
	public static FileHashMap<String, ArrayList<Integer>> loadBodiesSentimentsAsHashFile(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, ArrayList<Integer>> bSent = new FileHashMap<String, ArrayList<Integer>>(hashFileName,
				FileHashMap.RECLAIM_FILE_GAPS);
		return bSent;
	}

}
