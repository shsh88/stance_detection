package ude.master.thesis.stance_detection.analsys;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.Set;

import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;

public class ArgumentativeBodiesAnalysis {

	
	
	private static List<List<String>> trainingStances;
	private static List<List<String>> testStances;
	private static Map<Integer, String> trainingBodies;
	private static HashMap<Integer, String> testBodies;
	StanfordCoreNLP pipeline;
	

	public void loadData() throws IOException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.ARGUMENTED_BODIES_TRAIN,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.ARGUMENTED_BODIES_TEST);

		trainingStances = sddr.getTrainStances();
		trainingBodies = sddr.getTrainIdBodyMap();

		testStances = sddr.getTestStances();
		testBodies = sddr.getTestIdBodyMap();
	}
	public void count(List<List<String>> stances, Map<Integer, String> idBoyMap, String dataType) {
		
		int lessThan10 = 0;
		List<String> lessThan10Data = new ArrayList<>();
		Set<String> lessThan10Ids = new HashSet<>();
		
		int equals9 = 0;
		List<String> equals9Data = new ArrayList<>();
		Set<String> equals9Ids = new HashSet<>();
		
		int equals8 = 0;
		List<String> equals8Data = new ArrayList<>();
		Set<String> equals8Ids = new HashSet<>();
		
		int equals7 = 0;
		List<String> equals7Data = new ArrayList<>();
		Set<String> equals7Ids = new HashSet<>();
		
		int equals6 = 0;
		List<String> equals6Data = new ArrayList<>();
		Set<String> equals6Ids = new HashSet<>();

		int lessThan5 = 0;
		List<String> lessThan5Data = new ArrayList<>();
		Set<String> lessThan5Ids = new HashSet<>();

		int lessThan2 = 0;
		List<String> lessThan2Data = new ArrayList<>();
		Set<String> lessThan2Ids = new HashSet<>();
		
		int noArg = 0;
		List<String> noArgData = new ArrayList<>();
		Set<String> noArgIds = new HashSet<>();


		for (List<String> s : stances) {
			String body = idBoyMap.get(Integer.valueOf(s.get(1)));

			int c = countSentences(body);

			if (c < 10) {
				lessThan10++;
				lessThan10Data.add(s.get(0) + " , " + s.get(1) + " , " + s.get(2));
				lessThan10Ids.add(s.get(1));
			}
			if (c == 9) {
				equals9++;
				equals9Data.add(s.get(0) + " , " + s.get(1) + " , " + s.get(2));
				equals9Ids.add(s.get(1));
			}
			if (c == 8) {
				equals8++;
				equals8Data.add(s.get(0) + " , " + s.get(1) + " , " + s.get(2));
				equals8Ids.add(s.get(1));

			}
			if (c == 7) {
				equals7++;
				equals7Data.add(s.get(0) + " , " + s.get(1) + " , " + s.get(2));
				equals7Ids.add(s.get(1));
			}
			if (c == 6) {
				equals6++;
				equals6Data.add(s.get(0) + " , " + s.get(1) + " , " + s.get(2));
				equals6Ids.add(s.get(1));
			}
			if (c <= 5) {
				lessThan5++;
				lessThan5Data.add(s.get(0) + " , " + s.get(1) + " , " + s.get(2));
				lessThan5Ids.add(s.get(1));
			}
			if (c <= 2) {
				lessThan2++;
				lessThan2Data.add(s.get(0) + " , " + s.get(1) + " , " + s.get(2));
				lessThan2Ids.add(s.get(1));
			}
			
			if (c == 0) {
				noArg++;
				noArgData.add(s.get(0) + " , " + s.get(1) + " , " + s.get(2));
				noArgIds.add(s.get(1));
			}

		}

		if (dataType.equals("train")) {
			writeInfo(10, lessThan10, lessThan10Data, lessThan10Ids, ProjectPaths.TRAIN_LESS_10_ARGBODY);
			writeInfo(5, lessThan5, lessThan5Data, lessThan5Ids, ProjectPaths.TRAIN_LESS_5_ARGBODY);
			writeInfo(2, lessThan2, lessThan2Data, lessThan2Ids, ProjectPaths.TRAIN_LESS_2_ARGBODY);
			writeInfo(9, equals9, equals9Data, equals9Ids, ProjectPaths.TRAIN_eq9_ARGBODY);
			writeInfo(8, equals8, equals8Data, equals8Ids, ProjectPaths.TRAIN_eq8_ARGBODY);
			writeInfo(7, equals7, equals7Data, equals7Ids, ProjectPaths.TRAIN_eq7_ARGBODY);
			writeInfo(6, equals6, equals6Data, equals6Ids, ProjectPaths.TRAIN_eq6_ARGBODY);
			writeInfo(0, noArg, noArgData, noArgIds, ProjectPaths.TRAIN_0ARG_ARGBODY);

		} else {
			writeInfo(10, lessThan10, lessThan10Data, lessThan10Ids, ProjectPaths.TEST_LESS_10_ARGBODY);
			writeInfo(5, lessThan5, lessThan5Data, lessThan5Ids, ProjectPaths.TEST_LESS_5_ARGBODY);
			writeInfo(2, lessThan2, lessThan2Data, lessThan2Ids, ProjectPaths.TEST_LESS_2_ARGBODY);
			writeInfo(9, equals9, equals9Data, equals9Ids, ProjectPaths.TEST_eq9_ARGBODY);
			writeInfo(8, equals8, equals8Data, equals8Ids, ProjectPaths.TEST_eq8_ARGBODY);
			writeInfo(7, equals7, equals7Data, equals7Ids, ProjectPaths.TEST_eq7_ARGBODY);
			writeInfo(6, equals6, equals6Data, equals6Ids, ProjectPaths.TEST_eq6_ARGBODY);
			writeInfo(0, noArg, noArgData, noArgIds, ProjectPaths.TEST_0ARG_ARGBODY);

		}
	}

	private void writeInfo(int lessThancount, int count, List<String> instances, Set<String> lessThanIds, String path) {
		Writer writer = null;

		try {
			writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path), "utf-8"));
			writer.write("Bodies that contain less than" + lessThancount + "sentences: " + count + "\n");

			writer.write("Bodies' ids: " + lessThanIds);
			for (String ins : instances) {
				writer.write(ins + "\n");
			}
		} catch (IOException ex) {
			// report
		} finally {
			try {
				writer.close();
			} catch (Exception ex) {
				/* ignore */}
		}
	}
	
	private int countSentences(String text) {
		if (pipeline == null)
			pipeline = getStanfordPipeline();

		Annotation doc = new Annotation(text);
		pipeline.annotate(doc);
		List<CoreMap> sentences = doc.get(SentencesAnnotation.class);

		return sentences.size();
	}
	
	public double getAvgSentCount(Map<Integer, String> idBodyMap){
		int c = 0;
		for(Entry<Integer, String> e : idBodyMap.entrySet()){
			c += countSentences(e.getValue());
		}
		return c/idBodyMap.size();
	}
	
	public double getAvgSentCountStances(Map<Integer, String> idBodyMap, List<List<String>> stances){
		int c = 0;
		for(List<String> s : stances){
			int bodyId = Integer.valueOf(s.get(1));
			String body = idBodyMap.get(bodyId);
			c += countSentences(body);
		}
		return c/stances.size();
	}

	private StanfordCoreNLP getStanfordPipeline() {
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize,ssplit,pos,lemma");
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		return pipeline;
	}

	public static void main(String[] args) throws IOException {
		ArgumentativeBodiesAnalysis x = new ArgumentativeBodiesAnalysis();
		x.loadData();
		//x.count(trainingStances, trainingBodies, "train");
		//x.count(testStances, testBodies, "test");
		
		//System.out.println(x.getAvgSentCount(trainingBodies));
		//System.out.println(x.getAvgSentCount(testBodies));
		
		System.out.println(x.getAvgSentCountStances(trainingBodies, trainingStances));
		System.out.println(x.getAvgSentCountStances(testBodies, testStances));
	}
}
