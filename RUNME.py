# Databricks notebook source
# MAGIC %md This notebook sets up the companion cluster(s) to run the solution accelerator. It also creates the Workflow to illustrate the order of execution. Happy exploring! 
# MAGIC 🎉
# MAGIC
# MAGIC **Steps**
# MAGIC 1. Simply attach this notebook to a cluster and hit Run-All for this notebook. A multi-step job and the clusters used in the job will be created for you and hyperlinks are printed on the last block of the notebook. 
# MAGIC
# MAGIC 2. Run the accelerator notebooks: Feel free to explore the multi-step job page and **run the Workflow**, or **run the notebooks interactively** with the cluster to see how this solution accelerator executes. 
# MAGIC
# MAGIC     2a. **Run the Workflow**: Navigate to the Workflow link and hit the `Run Now` 💥. 
# MAGIC   
# MAGIC     2b. **Run the notebooks interactively**: Attach the notebook with the cluster(s) created and execute as described in the `job_json['tasks']` below.
# MAGIC
# MAGIC **Prerequisites** 
# MAGIC 1. You need to have cluster creation permissions in this workspace.
# MAGIC
# MAGIC 2. In case the environment has cluster-policies that interfere with automated deployment, you may need to manually create the cluster in accordance with the workspace cluster policy. The `job_json` definition below still provides valuable information about the configuration these series of notebooks should run with. 
# MAGIC
# MAGIC **Notes**
# MAGIC 1. The pipelines, workflows and clusters created in this script are not user-specific. Keep in mind that rerunning this script again after modification resets them for other users too.
# MAGIC
# MAGIC 2. If the job execution fails, please confirm that you have set up other environment dependencies as specified in the accelerator notebooks. Accelerators may require the user to set up additional cloud infra or secrets to manage credentials. 

# COMMAND ----------

# DBTITLE 0,Install util packages
# MAGIC %pip install git+https://github.com/databricks-academy/dbacademy@v1.0.13 git+https://github.com/databricks-industry-solutions/notebook-solution-companion@safe-print-html --quiet --disable-pip-version-check
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from solacc.companion import NotebookSolutionCompanion
nsc = NotebookSolutionCompanion()
user_name = nsc.w.current_user.me().user_name

# COMMAND ----------

job_json = {
        "timeout_seconds": 57600,
        "max_concurrent_runs": 1,
        "tags": {
            "usage": "solacc_testing",
            "group": "RCG"
        },
        "tasks": [
            {
                "task_key": "data_download",
                "notebook_task": {
                    "notebook_path": "01-data-download"
                },
                "job_cluster_key": "review-sum-ml-cluster"
            },
            {
                "task_key": "explore_prep",
                "depends_on": [
                    {
                        "task_key": "data_download"
                    }
                ],
                "notebook_task": {
                    "notebook_path": "02-explore-prep"
                },
                "job_cluster_key": "review-sum-ml-cluster"
            },
            {
                "task_key": "prompt_engineering",
                "depends_on": [
                    {
                        "task_key": "explore_prep"
                    }
                ],
                "notebook_task": {
                    "notebook_path": "03-prompt-engineering"
                },
                "job_cluster_key": "review-sum-ml-gpu-cluster"
            },
            {
                "task_key": "summarisation",
                "depends_on": [
                    {
                        "task_key": "prompt_engineering"
                    }
                ],
                "notebook_task": {
                    "notebook_path": "04-summarisation"
                },
                "job_cluster_key": "review-sum-ml-gpu-cluster"
            },
            {
                "task_key": "condensation",
                "depends_on": [
                    {
                        "task_key": "summarisation"
                    }
                ],
                "notebook_task": {
                    "notebook_path": "05-condensation"
                },
                "job_cluster_key": "review-sum-ml-gpu-cluster"
            },
            {
                "task_key": "presentation",
                "depends_on": [
                    {
                        "task_key": "condensation"
                    }
                ],
                "notebook_task": {
                    "notebook_path": "06-presentation"
                },
                "job_cluster_key": "review-sum-ml-cluster"
            }
        ],
        "job_clusters": [
            {
                "job_cluster_key": "review-sum-ml-cluster",
                "new_cluster": {
                    "spark_version": "13.2.x-cpu-ml-scala2.12",
                    "spark_conf": {
                        "spark.databricks.delta.preview.enabled": "true"
                    },
                    "node_type_id": {"AWS": "i3.4xlarge", "MSA": "Standard_D16ds_v5"},
                    "enable_elastic_disk": True,
                    "runtime_engine": "STANDARD",
                    "num_workers": 8
                }
            },
            {
                "job_cluster_key": "review-sum-ml-gpu-cluster",
                "new_cluster": {
                    "spark_version": "13.2.x-gpu-ml-scala2.12",
                    "spark_conf": {
                        "spark.databricks.delta.preview.enabled": "true"
                    },
                    "node_type_id": {"AWS": "g5.4xlarge", "MSA": "Standard_NC4as_T4_v3"}, # or "Standard_NC24ads_A100_v4" on MSA for A100 if capacity is available - compatible VMs are not available on GCP at this time
                    "enable_elastic_disk": True,
                    "runtime_engine": "STANDARD",
                    "single_user_name": user_name,
                    "data_security_mode": "SINGLE_USER",
                    "num_workers": 3
                }
            }
        ]
    }

# COMMAND ----------

dbutils.widgets.dropdown("run_job", "False", ["True", "False"])
run_job = dbutils.widgets.get("run_job") == "True"
nsc.deploy_compute(job_json, run_job=run_job)

# COMMAND ----------


