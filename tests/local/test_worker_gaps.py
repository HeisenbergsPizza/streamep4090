import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import unittest

TOOLS = Path(__file__).resolve().parents[2] / "experiments" / "scripts" / "rtx4090"
sys.path.insert(0, str(TOOLS))
from analyze_worker_gaps import analyze


class WorkerGapTest(unittest.TestCase):
    def test_overlap_and_no_work_bound(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            database, choices = root/"events.sqlite", root/"choices.json"
            choices.write_text(json.dumps({"start_ns": 11000, "end_ns": 12000}))
            with sqlite3.connect(database) as db:
                db.execute("CREATE TABLE NVTX_EVENTS(start,end,text,textId)")
                db.execute("CREATE TABLE StringIds(id PRIMARY KEY,value)")
                db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start,end,deviceId,demangledName)")
                db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start,end,deviceId)")
                db.execute("INSERT INTO StringIds VALUES(1,'ncclDevKernel')")
                db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(1020,1040,0,1)")
                db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(1030,1045,0)")
                # In trace units origin=1000, interval [1000,1100).
                marks = [(900,None,"WG.clock=10900"), (1200,None,"WG.clock=11200"),
                         (970,None,"WG.ready=0"), (1010,None,"WG.ready=1"),
                         (1011,None,"WG.wake=top8:1"), (1015,None,"WG.claim=4"),
                         (1016,None,"WG.ready=0"), (1017,1025,"WG.work=4"),
                         (1050,None,"WG.retire=4"), (1090,None,"WG.ready=1"),
                         (1091,None,"WG.wake=expert_input:2")]
                db.executemany("INSERT INTO NVTX_EVENTS VALUES(?,?,?,NULL)", marks)
            r=analyze(database,choices,[0,1e-7],None)
            self.assertEqual(r["gpu"]["active_ns"],25)
            self.assertEqual(r["idle_ns"],75)
            self.assertEqual(r["queue_empty_without_observed_claim_ns"],50)
            self.assertEqual(sum(b["idle_ns"] for b in r["time_bins"]),75)
            self.assertEqual(sum(b["idle_empty_no_claim_ns"] for b in r["time_bins"]),50)
            self.assertEqual(r["queue_empty_without_observed_claim_ns_by_next_wake"],
                             {"top8":10,"expert_input":40})
            self.assertEqual(r["idle_partition_ns"]["queue_empty"]["issued_before_host_retire"],5)
            self.assertEqual(sum(r["largest_idle_gaps"][0]["state_ns"].values()),55)

    def test_finalize_while_previous_running(self):
        from analyze_worker_gaps import held_intervals
        events = dict(claim=[(0,0,12),(2,2,13),(7,7,14)], work=[(1,2,12),(3,4,13),(8,10,14)],
                      finalize=[(4,4,13)],retire=[(4,4,13),(5,5,12),(13,13,14)])
        held,counts=held_intervals(events,0,12)
        self.assertEqual(counts["matched_intersecting_window"],3)
        self.assertEqual(held["finalize_guard"],[(4,13)])


if __name__ == "__main__":
    unittest.main()
