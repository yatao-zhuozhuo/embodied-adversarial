// Submit and inspect baseline tasks using the user's local YiCloud Go SDK.
package main

import (
 "encoding/json"
 "errors"
 "flag"
 "fmt"
 "os"
 "time"
 "gitee.com/yicloud-team/openapi-sdk-go/base"
 "gitee.com/yicloud-team/openapi-sdk-go/base/auth"
 "gitee.com/yicloud-team/openapi-sdk-go/base/config"
 "gitee.com/yicloud-team/openapi-sdk-go/base/msgs"
 "gitee.com/yicloud-team/openapi-sdk-go/services/job"
)

func main() { if err:=run(); err!=nil {fmt.Fprintln(os.Stderr,err);os.Exit(1)} }
func run() error {
 spec:=flag.String("spec","","CreateJobReq JSON path")
 receipt:=flag.String("receipt","","durable submission receipt")
 inspect:=flag.Bool("inspect",false,"read current job state without submission")
 dry:=flag.Bool("dry-run",false,"validate request locally without API call")
 flag.Parse()
 if *spec=="" {return errors.New("-spec is required")}
 raw,err:=os.ReadFile(*spec);if err!=nil{return err}
 var req job.CreateJobReq
 if err=json.Unmarshal(raw,&req);err!=nil{return err}
 if req.Name==nil || req.Project==nil || req.JobId==nil || len(req.TaskSpecs)!=1 {return errors.New("require Name, Project, explicit JobId and one TaskSpec")}
 for _,t:=range req.TaskSpecs {if t.Image==nil || t.Replicas==nil || *t.Replicas!=1 || len(t.Command)==0 {return errors.New("invalid single-replica task")}}
 if *dry {fmt.Printf("validated: %s / %s\n",*req.Project,*req.JobId);return nil}
 credential:=auth.NewCredential()
 if credential.PublicKey=="" || credential.PrivateKey=="" {return errors.New("YICLOUD_PUBLIC_KEY and YICLOUD_SECRET_KEY must be injected; no API request was made")}
 cfg:=config.New();cfg.Timeout=60*time.Second;cfg.MaxRetries=0
 job.UseClient(base.NewClient(base.WithConfig(cfg),base.WithCredential(credential)))
 ctx,_:=msgs.NewMetaCtx()
 if *inspect {
  d,e:=job.GetJob(ctx,&job.GetJobReq{Project:req.Project,JobId:req.JobId});if e!=nil{return e}
  return json.NewEncoder(os.Stdout).Encode(map[string]any{"job_id":d.JobId,"phase":d.Phase,"message":d.Message,"project":d.Project})
 }
 if *receipt=="" {return errors.New("-receipt required for create")}
 if _,e:=os.Stat(*receipt);e==nil {return errors.New("receipt already exists; use -inspect rather than resubmitting")}
 // Explicit ID makes an ambiguous timeout inspectable; do not blindly retry creates.
 d,createErr:=job.CreateJob(ctx,&req);if createErr!=nil{return fmt.Errorf("create failed or outcome unknown; inspect explicit JobId %s before retry: %w",*req.JobId,createErr)}
 out,e:=json.MarshalIndent(d,"","  ");if e!=nil{return e}
 f,e:=os.OpenFile(*receipt,os.O_WRONLY|os.O_CREATE|os.O_EXCL,0600);if e!=nil{return e}
 defer f.Close();if _,e=f.Write(append(out,'\n'));e!=nil{return e}
 fmt.Println(string(out));return nil
}
