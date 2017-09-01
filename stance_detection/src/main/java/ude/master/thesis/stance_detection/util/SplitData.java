package ude.master.thesis.stance_detection.util;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;

public class SplitData {

	public static void main(String[] args) {
		removeUnrelated();
		makeRelatedUnrelatedData();
	}

	private static void makeRelatedUnrelatedData() {
		CSVReader reader = null;
		CSVWriter writer = null;
		try {
			//reader = new CSVReader(new FileReader("resources/data/train_stances.csv"));
			reader = new CSVReader(new FileReader("resources/data/test_data/competition_test_stances.csv"));
			writer = new CSVWriter(new FileWriter("resources/data/test_data/competition_test_stances_related_unrelated.csv"));
			String[] line;
			line = reader.readNext();
			writer.writeNext(line);
			while ((line = reader.readNext()) != null) {
				if(!line[2].equals("unrelated"))
					line[2] = "related";
				writer.writeNext(line);
			}
			reader.close();
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}

	private static void removeUnrelated() {
		CSVReader reader = null;
		CSVWriter writer = null;
		try {
			reader = new CSVReader(new FileReader("resources/data/test_data/competition_test_stances.csv"));
			writer = new CSVWriter(new FileWriter("resources/data/test_data/competition_test_stances_no_unrelated.csv"));
			String[] line;
			line = reader.readNext();
			writer.writeNext(line);
			while ((line = reader.readNext()) != null) {
				if(!line[2].equals("unrelated"))
					writer.writeNext(line);
			}
			reader.close();
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
