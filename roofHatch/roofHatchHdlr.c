// DATA TYPES
typedef struct {
   RoofHatch_T                     RoofHatch_Command;
} RoofHatchHdlr;

// PUBLIC FUNCTIONS
FUNC(void, RTE_ROOFHATCHHDLR_APPL_CODE) RoofHatch_Hdlr_Init(void)
{
   
}

FUNC(void, RTE_ROOFHATCHHDLR_APPL_CODE) RoofHatch_Hdlr_run(void)
{

   RoofHatchHdlr            Handler_obj;
   
   Handler_obj.RoofHatch_Command = Rte_Read_RoofHatch_Command_RoofHatch_Command(&Handler_obj.RoofHatch_Command);
   
   if (Handler_obj.RoofHatch_Command == OPEN)
   {
        (void)Rte_Call_RoofHatchHdlr_Actuator_setOpen(IOHW_ON);
        (void)Rte_Call_RoofHatchHdlr_Actuator_setClose(IOHW_OFF);
   }
   else if(Handler_obj.RoofHatch_Command == CLOSE)
   {
        (void)Rte_Call_RoofHatchHdlr_Actuator_setOpen(IOHW_OFF);
        (void)Rte_Call_RoofHatchHdlr_Actuator_setClose(IOHW_ON);
   }
   else
   {
       (void)Rte_Call_RoofHatchHdlr_Actuator_setOpen(IOHW_OFF);
       (void)Rte_Call_RoofHatchHdlr_Actuator_setClose(IOHW_OFF);
   }

}
