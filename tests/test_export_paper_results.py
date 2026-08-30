import inspect
import unittest
import scripts.export_paper_results as exporter


class PaperResultsExportTest(unittest.TestCase):
    def test_main_method_inventory_is_frozen(self): self.assertEqual(exporter.MAIN,("NVP","MLP","LSTM","Transformer","Mamba","Direct MambaNVP","Anchored MambaNVP","PA-MambaNVP"))
    def test_exporter_is_report_only(self):
        source=inspect.getsource(exporter); self.assertNotIn("import compiler_gym",source); self.assertNotIn("torch",source); self.assertNotIn("subprocess",source)


if __name__=="__main__": unittest.main()
