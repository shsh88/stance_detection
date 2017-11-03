package ude.master.thesis.stance_detection.analsys;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.opencsv.CSVReader;

import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;

public class DiscussNonDiscussErrorAnalysis {

	private static Map<Integer, String> trainIdBodyMap = new HashMap<Integer, String>();
	private static List<List<String>> trainingStances = new ArrayList<>();
	private static HashMap<Integer, String> testIdBodyMap = new HashMap<>();
	private static List<List<String>> testStances = new ArrayList<List<String>>();
	private static HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap = new HashMap<>();
	private static HashMap<Integer, Map<Integer, String>> testSummIdBoyMap = new HashMap<>();

	public static void loadData() throws IOException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true, ProjectPaths.TRAIN_STANCES,
				ProjectPaths.SUMMARIZED_TRAIN_BODIES, ProjectPaths.TEST_STANCES_NO_UNRELATED, ProjectPaths.SUMMARIZED_TEST_BODIES);

		trainingSummIdBoyMap = sddr.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TRAIN_BODIES));
		testSummIdBoyMap = sddr.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TEST_BODIES));

		trainingStances = sddr.getTrainStances();

		testStances = sddr.getTestStances();
	}

	public static void main(String[] args) throws IOException {
		loadData();

		CSVReader reader = null;
		reader = new CSVReader(new FileReader("C:/thesis_stuff/analysis/Three_Step_BoW1000_1000f_10-30_15-14_with_test_.csv"));
		String[] line;
		line = reader.readNext();
		int i = 0;

		List<List<String>> falseClassified = new ArrayList<>();
		while ((line = reader.readNext()) != null) {
			//System.out.println(Arrays.toString(line));
			if (line[3].equals("+")&& line[1].contains("non_discuss")) {
				List<String> toAdd = new ArrayList<>();
				List<String> wrongInstance = testStances.get(Integer.valueOf(line[0]) - 1);
				toAdd.addAll(wrongInstance);
				toAdd.add(testSummIdBoyMap.get(Integer.valueOf(wrongInstance.get(1))).get(1));
				toAdd.add(testSummIdBoyMap.get(Integer.valueOf(wrongInstance.get(1))).get(2));
				toAdd.add(testSummIdBoyMap.get(Integer.valueOf(wrongInstance.get(1))).get(3));
				falseClassified.add(toAdd);
		
			}
			i++;
		}
		reader.close();

		Writer writer = null;

		try {
			writer = new BufferedWriter(new OutputStreamWriter(
					new FileOutputStream("C:/thesis_stuff/analysis/Three_Step_BoW1000_1000f_10-30_15-14_with_test_analysis2.txt"),
					"utf-8"));

			for (List<String> f : falseClassified) {
				writer.write(f + "\n");
				writer.write("\n");
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

}
