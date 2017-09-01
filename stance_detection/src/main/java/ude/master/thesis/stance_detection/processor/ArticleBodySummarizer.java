package ude.master.thesis.stance_detection.processor;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import com.opencsv.CSVWriter;

import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;

public class ArticleBodySummarizer {

	public static String summarize(String body) {
		String summBody;
		if (body.length() > 1000) {

			int periodPos = body.indexOf('.');

			if (periodPos == -1) {
				periodPos = body.indexOf(' ', 256);
				summBody = body.substring(0, periodPos + 1);
				// summBody += body.substring(body.indexOf(' ',
				// body.length()-300));
			} else {
				while (periodPos < 256) {
					periodPos = body.indexOf('.', periodPos + 1);
				}
				summBody = body.substring(0, periodPos + 1);
			}

			periodPos = body.indexOf('.', body.length() - 400);
			if (periodPos == -1) {
				periodPos = body.indexOf(' ', body.length() - 300);
				summBody = body.substring(periodPos);
			} else {
				int shift = 100;
				while (periodPos > (body.length() - 200)) {
					periodPos = body.lastIndexOf('.', periodPos - shift);
				}
				summBody += body.substring(periodPos + 1);
			}
			body = summBody;
		}
		return body;
	}

	public static void main(String[] args) throws IOException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true);
		Map<Integer, String> trainBodyIdMap = sddr.getTrainIdBodyMap();
		HashMap<Integer, String> testBodyIdMap = sddr.getTestIdBodyMap();

		summarizeFileAndSave(trainBodyIdMap, "resources/data/summ_train_bodies.csv");
		summarizeFileAndSave(testBodyIdMap, "resources/data/test_data/summ_competition_test_bodies.csv");
	}

	private static void summarizeFileAndSave(Map<Integer, String> trainBodyIdMap, String filename) throws IOException {
		List<String[]> entries = new ArrayList<>();
		entries.add(new String[] { "Body ID", "articleBody" });

		int i = 0;
		for (Map.Entry<Integer, String> e : trainBodyIdMap.entrySet()) {

			List<String> entry = new ArrayList<>();

			String summBody = summarize(e.getValue());

			entry.add(Integer.toString(e.getKey()));
			entry.add(summBody);

			entries.add(entry.toArray(new String[entry.size()]));
			i++;

			//if (i == 20)
			//	break;
			if (i % 1000 == 0)
				System.out.println("processed : " + i);
		}

		try (CSVWriter writer = new CSVWriter(new FileWriter(filename))) {
			writer.writeAll(entries);
		}
	}

}
