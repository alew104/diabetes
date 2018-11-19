import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

public class ParseXML {

	public static void main(String[] args)
			throws SAXException, IOException, ParserConfigurationException {
		// TODO Auto-generated method stub
		ArrayList<File> xmlFiles = addXML();
		ArrayList<File> xmlPlainFiles = addPlainXML();

		for (int i = 0; i < xmlFiles.size(); i++) {
			String file = xmlFiles.get(i).getName();
			PrintWriter pw = new PrintWriter("out/" + file.substring(0, file.length() - 3) + "txt", "UTF-8");
			parseLine(xmlFiles.get(i), pw);
		}
		
		for (int i = 0; i < xmlPlainFiles.size(); i++) {
			String file = xmlPlainFiles.get(i).getName();
			PrintWriter pw = new PrintWriter("out/" + file.substring(0, file.length() - 3) + "txt", "UTF-8");
			parseLine(xmlPlainFiles.get(i), pw);
		}

		/*
		 * String img = "01"; String subXml = "01"; File f = new
		 * File("xml/diaretdb1_image0" + img + "_" + subXml + ".xml");
		 * PrintWriter pw = new PrintWriter("out/derp.txt", "UTF-8");
		 * parseLine(f, pw);
		 */
	}

	public static void parseLine(File in, PrintWriter out)
			throws SAXException, IOException, ParserConfigurationException {
		DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
		DocumentBuilder db = dbf.newDocumentBuilder();
		Document doc = db.parse(in);
		NodeList list = doc.getElementsByTagName("*");
		for (int i = 0; i < list.getLength(); i++) {
			// Get element
			Node element = (Element) list.item(i);
			if (!element.getNodeName().equals("marking")
					&& !element.getNodeName().equals("imgannotooldata")
					&& !element.getNodeName().equals("header")
					&& !element.getNodeName().equals("creator")
					&& !element.getNodeName().equals("software")
					&& !element.getNodeName().equals("affiliation")
					&& !element.getNodeName().equals("copyrightnotice")
					&& !element.getNodeName().equals("markinglist")
					&& !element.getNodeName().equals("centroid")) {
				
				if (element.getNodeName().equals("coords2d") || element.getNodeName().equals("markingtype")) {
					String line = element.getNodeName() + ","
							+ element.getTextContent();
					out.println(line);
				}
			}
		}
		out.close();
	}

	public static ArrayList<File> addXML() {
		String img;
		String subXml;
		ArrayList<File> a = new ArrayList<>();
		for (int i = 1; i < 90; i++) {
			if (i < 10) {
				img = "0" + i;
			} else {
				img = "" + i;
			}
			for (int j = 1; j < 5; j++) {
				subXml = "0" + j;
				a.add(new File(
						"xml/diaretdb1_image0" + img + "_" + subXml + ".xml"));
			}
		}
		return a;
	}

	public static ArrayList<File> addPlainXML() {
		String img;
		String subXml;
		ArrayList<File> a = new ArrayList<>();
		for (int i = 1; i < 90; i++) {
			if (i < 10) {
				// when number is less than 10 use pad with a 0
				img = "0" + i;
			} else {
				img = "" + i;
			}
			for (int j = 1; j < 5; j++) {
				subXml = "0" + j;
				a.add(new File("xml/diaretdb1_image0" + img + "_" + subXml
						+ "_plain.xml"));
			}
		}
		return a;
	}

}
