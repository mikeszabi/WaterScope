# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:21:47 2017

@author: SzMike
"""

#https://docs.python.org/2/library/xml.etree.elementtree.html#modifying-an-xml-file

from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
 

class XMLWriter:

    def __init__(self, author='SU'):
        self.top = Element('measure')
        self.measuredvolume = SubElement(self.top,'measuredvolume_in_liter')

        self.results = SubElement(self.top, 'results')

        self.scaledresults = SubElement(self.top, 'scaledresults')
        
        self.allobjectnumber_results = SubElement(self.results, 'allobjectnumber')
        self.allobjectnumber_scaledresults = SubElement(self.scaledresults, 'allobjectnumber')
                
        self.objectsbytype_results = SubElement(self.results, 'objectsbytype')
        self.objectsbytype_scaledresults = SubElement(self.scaledresults, 'objectsbytype')

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True)

    def addMeasuredVolume(self, measured_volume):
        assert isinstance(measured_volume,float), "Not valid measured_volume"
        self.measuredvolume.text=str(measured_volume)
        
    def addTaxonStat(self, taxon_name, count, scaled=False):
        assert isinstance(taxon_name,str), "Not valid taxon name"
        assert isinstance(count,int), "Not valid taxon count"
        
        if scaled:
            objectsbytype=self.objectsbytype_scaledresults
        else:
            objectsbytype=self.objectsbytype_results
              
        classes=taxon_name.split('.')
        
        objectnumber = SubElement(objectsbytype, 'objectnumber')
        for i,cl in enumerate(classes):
            objectnumber.set('class'+str(i+1),cl)
#        for i,cl in enumerate(classes[::-1]):
#            objectnumber.set('class'+str(len(classes)-i),cl)

        objectnumber.text=str(count)
        
    def addAllCount(self, count, scaled):
        assert isinstance(count,int), "Not valid taxon count"
        if scaled:
            allobjectnumber=self.allobjectnumber_scaledresults
        else:
            allobjectnumber=self.allobjectnumber_results

        allobjectnumber.text = str(count)


    def save(self, targetFile=None):
        out_file = None
        if targetFile is None:
            out_file = open(self.filename + '.xml', 'w')
        else:
            out_file = open(targetFile, 'w')

        prettifyResult = self.prettify(self.top)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()


class XMLReader:

    def __init__(self, filepath):
        self.shapes = []
        self.filepath = filepath
        self.parseXML()

    def parseXML(self,ps='results/objectsbytype/objectnumber'):
        assert self.filepath.endswith('.xml'), "Unsupport file format"
        parser = etree.XMLParser(encoding='utf-8')
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        #filename = xmltree.find('filename').text

        for class_name in xmltree.findall(ps):
            # one polygon per shape
           print(class_name.attrib['class1'])
        return True
"""
filepath=r'd:\Projects\WaterScope\work_0\Measurement\20170712\002\MeasureSum_20170712_1455.xml'
filepath=r'd:\Projects\WaterScope\work_0\Measurement\20170712\alma.xml'
tempParseReader = XMLReader(filepath)
# print tempParseReader.getShapes()

# Test
filepath=r'd:\Projects\WaterScope\work_0\Measurement\20170712\alma.xml'
tmp = XMLWriter()
tmp.addTaxonStat('alma.korte.barack',10,False)
tmp.addMeasuredVolume(0.001)
tmp.save(targetFile=filepath)
"""
