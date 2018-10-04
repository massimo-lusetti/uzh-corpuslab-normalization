./Main-wus-soft-train-pos.sh norm_soft_pos wus/phase2/treetagger wus
./Main-wus-soft-train-pos.sh norm_soft_context wus/phase2/treetagger wus
./Main-wus-soft-train-pos.sh norm_soft_context wus/phase2/treetagger wus aux
./Main-wus-soft-train-pos.sh norm_soft_char_context wus/phase2/treetagger wus
./Main-wus-soft-train-pos.sh norm_soft_char_context wus/phase2/treetagger wus aux

./Main-wus-soft-train-pos.sh norm_soft_pos wus/phase2/btagger wus_bt
./Main-wus-soft-train-pos.sh norm_soft_context wus/phase2/btagger wus_bt aux
./Main-wus-soft-train-pos.sh norm_soft_char_context wus/phase2/btagger wus_bt aux

./Main-wus-soft-train-pos.sh norm_soft_pos wus/phase2/btagger-sms wus_bt_sms
./Main-wus-soft-train-pos.sh norm_soft_context wus/phase2/btagger-sms wus_bt_sms aux
./Main-wus-soft-train-pos.sh norm_soft_char_context wus/phase2/btagger-sms wus_bt_sms aux
