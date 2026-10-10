# services/cases — M5 (Eya), ASP.NET Core + PostgreSQL + QuestPDF

**Branch:** `services/cases`

**Deliverable:** the sample, result, correction, validation and PDF report endpoints. Contracts: see the
"Cases service" section of [`contracts/README.md`](../../contracts/README.md). Database: [`contracts/db_schema.sql`](../../contracts/db_schema.sql).

**Until H7:**
- seed your database or in-memory store from `fakes/json/samples.json`, which has a sample for every status;
- call the mock inference server (`fakes/mock_inference/`, port 8001) for real HTTP round trips.

**At integration:** change the inference base URL from the mock (`http://localhost:8001`) to the real service.
